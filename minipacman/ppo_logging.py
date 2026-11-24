"""rsl_rl PPO that also reports the mean KL and the mean learning rate of each update.

TensorBoard entries added (rsl_rl's logger writes every loss_dict key as Loss/<key>):
  * Loss/mean_kl: the KL that PPO.update() computes anyway (one value per minibatch, measured before that minibatch's
    gradient step), averaged over the update. Only the adaptive schedule computes it, so with schedule="fixed" there
    is no mean_kl entry.
  * Loss/lr_mean: the learning rate of every gradient step in the update, averaged. rsl_rl's own Loss/learning_rate
    is only the value after the last minibatch; the adaptive schedule changes the rate every minibatch, so inside one
    update it can climb from 1e-5 to 1e-4 and fall back to 1e-5 by the end, and Loss/learning_rate shows only 1e-5.

Training is unchanged: nothing extra is computed from or fed back into the networks, the optimizer or the random
number generator.
"""
from __future__ import annotations

from rsl_rl.algorithms import PPO


class KLLoggingPPO(PPO):
    def update(self) -> dict[str, float]:
        kls: list[float] = []
        lrs: list[float] = []
        compute_kl = self.actor.get_kl_divergence

        def recording_kl(old_params, new_params):
            kl = compute_kl(old_params, new_params)
            kls.append(float(kl.mean()))
            return kl

        def record_lr(optimizer, args, kwargs):              # runs just before every optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        self.actor.get_kl_divergence = recording_kl          # instance attribute shadows the method
        lr_hook = self.optimizer.register_step_pre_hook(record_lr)
        try:
            loss_dict = super().update()
        finally:
            del self.actor.get_kl_divergence                  # back to the class method
            lr_hook.remove()
        if kls:
            loss_dict["mean_kl"] = sum(kls) / len(kls)
        if lrs:
            loss_dict["lr_mean"] = sum(lrs) / len(lrs)
        return loss_dict
