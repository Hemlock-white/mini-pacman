"""下載固定版本的 TITA 模型到 third_party/tita/。【你們實作】

為什麼不把模型放進 repo：DDTRobot/ddt_ros2_control 的 tita_description 沒有授權（package.xml 寫 TODO），
不能重新散布，所以用腳本從官方 repo 下載固定的 commit。third_party/ 已經在 .gitignore。

用法：
    python scripts/download_tita.py

要做的事（四個 git 指令，參數在 minipacman.config.TitaSource）：
    git init <target_dir>
    git -C <target_dir> remote add origin <repo_url>
    git -C <target_dir> fetch --depth 1 origin <commit>
    git -C <target_dir> checkout FETCH_HEAD
用 subprocess.run([...], check=True) 執行。已經下載過（<target_dir>/<subdir>/mujoco/robot.xml 存在）就直接結束。
最後印出模型路徑，並檢查 robot.xml 存在。
加分：用 sparse checkout 只下載 urdfs/tita_description（git sparse-checkout set <subdir>），比較快。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minipacman.config import TitaSource  # noqa: E402


def main() -> None:
    src = TitaSource()
    raise NotImplementedError("TODO：download_tita.py（見檔頭）")


if __name__ == "__main__":
    main()
