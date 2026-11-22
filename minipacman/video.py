"""Off-screen MuJoCo recordings for selected evaluation seeds.

Numeric experiments intentionally do not render every episode.  This module reruns a small,
explicit list of seeds after evaluation and records comparison videos with the same conditions.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .evaluate import Condition


@dataclass(frozen=True)
class VideoGroup:
    """Conditions shown together in one comparison video."""

    name: str
    conditions: tuple[Condition, ...]


PANEL_COLORS = (
    (245, 105, 80),
    (70, 210, 150),
    (255, 185, 70),
    (175, 120, 255),
)


def comparison_groups(experiment: str, conditions: list[Condition]) -> list[VideoGroup]:
    """Group experiment conditions into useful side-by-side videos."""
    if experiment == "exp1":
        rewards = list(dict.fromkeys(condition.reward_variant for condition in conditions))
        return [
            VideoGroup(
                f"exp1_{reward}",
                tuple(condition for condition in conditions if condition.reward_variant == reward),
            )
            for reward in rewards
        ]
    if experiment == "exp2":
        return [VideoGroup("exp2_filters", tuple(conditions))]
    if experiment == "exp3":
        return [VideoGroup("exp3_sources", tuple(conditions))]
    raise ValueError(f"video recording is not defined for {experiment!r}")


def tile_panels(panels: list[np.ndarray]) -> np.ndarray:
    """Tile 1–3 panels horizontally and four panels as a 2x2 grid."""
    if not panels:
        raise ValueError("at least one video panel is required")
    shape = panels[0].shape
    if any(panel.shape != shape for panel in panels):
        raise ValueError("all video panels must have the same shape")
    if len(panels) <= 3:
        return np.concatenate(panels, axis=1)
    if len(panels) == 4:
        return np.concatenate([
            np.concatenate(panels[:2], axis=1),
            np.concatenate(panels[2:], axis=1),
        ], axis=0)
    raise ValueError("comparison videos support at most four panels")


def record_experiment_videos(
    experiment: str,
    conditions: list[Condition],
    seeds: list[int],
    cfg: Config,
    output_dir: str | Path,
    device: str,
    fps: int = 25,
) -> list[Path]:
    """Record every comparison group for each selected seed."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for group in comparison_groups(experiment, conditions):
        for seed in seeds:
            path = output_dir / f"{group.name}_seed{int(seed)}.mp4"
            print(f"  [video] {path.name} ...", flush=True)
            summaries = record_comparison(
                list(group.conditions), int(seed), cfg, path, device=device, fps=fps,
            )
            paths.append(path)
            results = ", ".join(
                f"{summary['condition']}: success={summary['success']} "
                f"fell={summary['fell']} hits={summary['n_hit']}/{summary['n_thrown']}"
                for summary in summaries
            )
            print(f"  [video] done: {results}", flush=True)
    return paths


def record_comparison(
    conditions: list[Condition],
    seed: int,
    cfg: Config,
    output_path: str | Path,
    device: str,
    fps: int = 25,
    panel_width: int = 560,
    panel_height: int = 380,
) -> list[dict]:
    """Rerun one seed for 1–4 conditions and encode an annotated H.264 MP4."""
    if not 1 <= len(conditions) <= 4:
        raise ValueError("record_comparison requires one to four conditions")
    if fps <= 0:
        raise ValueError("fps must be positive")
    policy_rate = 1.0 / (cfg.sim.timestep * cfg.sim.policy_every)
    if fps > policy_rate + 1e-9:
        raise ValueError(f"fps cannot exceed the policy rate ({policy_rate:g} Hz)")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("video recording requires ffmpeg on PATH (Ubuntu: sudo apt install ffmpeg)")

    # Must be set before the first renderer context is created. Users can override it explicitly.
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import mujoco
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError("video recording requires MuJoCo and Pillow") from error

    from .dodge_env import DodgeEnv
    from .evaluate import ScriptedPolicy, load_policy

    policies = {}
    for condition in conditions:
        if condition.policy not in policies:
            policies[condition.policy] = (
                ScriptedPolicy(cfg) if condition.policy == "scripted"
                else load_policy(condition.policy, cfg, device)
            )
    envs = [
        DodgeEnv(
            cfg,
            reward_variant="task" if condition.reward_variant == "scripted" else condition.reward_variant,
            filter_mode=condition.filter_mode,
            replan=condition.replan,
            ball_source=condition.ball_source,
            augment_depth=False,
            seed=seed,
        )
        for condition in conditions
    ]
    observations = [env.reset(seed) for env in envs]
    renderers = [mujoco.Renderer(env.m, height=panel_height, width=panel_width) for env in envs]
    cameras = [_camera_for(mujoco, env) for env in envs]
    done = [False] * len(envs)
    flashes = [0] * len(envs)
    last_frame = None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = tile_panels([np.zeros((panel_height, panel_width, 3), dtype=np.uint8)] * len(envs))
    output_height, output_width = sample.shape[:2]
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{output_width}x{output_height}", "-r", str(fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    next_frame_time = 0.0
    max_steps = round(cfg.sim.episode_seconds / (cfg.sim.timestep * cfg.sim.policy_every))

    try:
        for step in range(max_steps):
            for index, (condition, env) in enumerate(zip(conditions, envs)):
                if done[index]:
                    continue
                action = policies[condition.policy].act(observations[index], env)
                observations[index], _, terminated, truncated, info = env.step(action)
                if info["new_hits"]:
                    flashes[index] = fps // 2
                done[index] = bool(terminated or truncated)
            simulation_time = max(float(env.d.time) for env in envs)
            if simulation_time + 1e-12 < next_frame_time:
                continue
            next_frame_time += 1.0 / fps

            panels = []
            for index, (condition, env, renderer, camera) in enumerate(
                zip(conditions, envs, renderers, cameras)
            ):
                _follow_camera(camera, env)
                renderer.update_scene(env.d, camera=camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                _add_goal_marker(mujoco, renderer, env)
                panels.append(_decorate(
                    Image, ImageDraw, ImageFont, renderer.render(), env,
                    _condition_label(condition), PANEL_COLORS[index], seed,
                    flashes[index], done[index], panel_width, panel_height,
                ))
                flashes[index] = max(0, flashes[index] - 1)
            last_frame = tile_panels(panels)
            if encoder.stdin is None:
                raise RuntimeError("ffmpeg stdin is unavailable")
            encoder.stdin.write(last_frame.tobytes())
            if all(done):
                break

        if last_frame is not None:
            if encoder.stdin is None:
                raise RuntimeError("ffmpeg stdin is unavailable")
            for _ in range(fps * 2):
                encoder.stdin.write(last_frame.tobytes())
    finally:
        for renderer in renderers:
            renderer.close()
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
        if return_code:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}")

    return [
        {"condition": condition.name, **env.episode_summary()}
        for condition, env in zip(conditions, envs)
    ]


def _condition_label(condition: Condition) -> str:
    if condition.name.startswith("exp1_"):
        mode = "filter off" if condition.filter_mode is None else "filter on"
        return f"{condition.reward_variant} | {mode}"
    if condition.name.startswith("exp2_"):
        mode = (condition.filter_mode or "filter off").replace("_", "-")
        return f"scripted | {mode}"
    if condition.name == "exp3_no_filter":
        return "filter off | truth"
    if condition.name.startswith("exp3_"):
        return f"{condition.ball_source.replace('_', ' ')} | lag-aware"
    if condition.filter_mode is None:
        return f"{condition.reward_variant} | filter off"
    source = condition.ball_source.replace("_", " ")
    mode = condition.filter_mode.replace("_", "-")
    return f"{source} | {mode}"


def _camera_for(mujoco, env):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.2
    camera.azimuth = 135
    camera.elevation = -18
    _follow_camera(camera, env)
    return camera


def _follow_camera(camera, env) -> None:
    base = env.d.xpos[env.ids.base_body]
    camera.lookat[:] = (float(base[0]) + 0.45, float(base[1]), 0.35)


def _add_goal_marker(mujoco, renderer, env) -> None:
    scene = renderer.scene
    rotation = np.eye(3, dtype=float).reshape(-1)
    x, y = map(float, env.goal)
    markers = (
        (mujoco.mjtGeom.mjGEOM_CYLINDER, (env.cfg.task.goal_radius, 0.025, 0.0), (x, y, 0.025),
         (0.1, 1.0, 0.2, 0.75)),
        (mujoco.mjtGeom.mjGEOM_CYLINDER, (0.025, 0.65, 0.0), (x, y, 0.65),
         (0.1, 1.0, 0.2, 1.0)),
        (mujoco.mjtGeom.mjGEOM_SPHERE, (0.13, 0.0, 0.0), (x, y, 1.32),
         (0.1, 1.0, 0.2, 1.0)),
    )
    for geom_type, size, position, rgba in markers:
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom], geom_type,
            np.asarray(size, dtype=float), np.asarray(position, dtype=float),
            rotation, np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1


def _decorate(
    Image,
    ImageDraw,
    ImageFont,
    frame: np.ndarray,
    env,
    title: str,
    color: tuple[int, int, int],
    seed: int,
    flash: int,
    done: bool,
    width: int,
    height: int,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=14)
    draw.rectangle((0, 0, width, 72), fill=(0, 0, 0, 178))
    draw.rectangle((0, 0, 8, height), fill=(*color, 255))
    draw.text((18, 9), title, font=font, fill=(*color, 255))
    hits = sum(bool(record.hit) for record in env.launcher.records)
    robot = env.env.planar_state(env.ref.s)
    distance = float(np.linalg.norm(env.goal - robot.xy))
    progress = float(np.clip(1.0 - distance / env.cfg.task.goal_distance, 0.0, 1.0))
    status = "FELL" if env.fell else (
        "GOAL REACHED" if env.reached_goal else ("FINISHED" if done else "RUNNING")
    )
    draw.text(
        (18, 40), f"seed {seed}  t {env.d.time:5.2f}s  hits {hits}  {status}",
        font=small, fill=(255, 255, 255, 255),
    )
    draw.text(
        (width - 194, 40), f"goal dist {distance:.2f}m",
        font=small, fill=(120, 255, 145, 255),
    )
    draw.rectangle((18, height - 22, width - 18, height - 10), fill=(0, 0, 0, 150))
    draw.rectangle(
        (18, height - 22, 18 + int((width - 36) * progress), height - 10),
        fill=(70, 225, 115, 230),
    )
    if flash > 0:
        draw.rectangle((8, 72, width - 1, height - 1), outline=(255, 55, 55, 255), width=9)
        draw.text((width - 95, 9), "BALL HIT", font=small, fill=(255, 80, 80, 255))
    return np.asarray(image)
