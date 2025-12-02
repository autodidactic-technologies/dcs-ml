import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


@dataclass
class ParamSpec:
    low: torch.Tensor  # shape [D]
    high: torch.Tensor  # shape [D]

    @property
    def dim(self) -> int:
        return int(self.low.numel())


def mlp(sizes, activation=nn.Tanh, output_activation=None):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1])]
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)


class ParamActorCritic(nn.Module):
    """
    Actor-Critic for parameterized action space with:
     - Discrete action selection among N branches
     - Continuous parameters per selected branch (dimensions vary per branch)

    For stability and easier bounded output, we sample raw Gaussian z, then apply tanh + affine map to bounds.
    We implement change-of-variables correction in log-prob.
    """
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, int], param_specs: List[ParamSpec], device: torch.device):
        super().__init__()
        self.obs_dim = obs_dim
        self.param_specs = param_specs
        self.n_actions = len(param_specs)
        self.device = device

        # Shared torso
        h1, h2 = hidden_sizes
        self.torso = mlp([obs_dim, h1, h2], activation=nn.Tanh)

        # Heads
        self.logits = nn.Linear(h2, self.n_actions)
        self.value_head = nn.Linear(h2, 1)

        # Per-branch parameter heads (mean and log_std)
        self.param_means = nn.ModuleList()
        self.param_log_stds = nn.ModuleList()
        for spec in param_specs:
            d = spec.dim
            self.param_means.append(nn.Linear(h2, d))
            # initialize near 0 std
            head = nn.Linear(h2, d)
            nn.init.constant_(head.weight, 0.0)
            nn.init.constant_(head.bias, -0.5)  # exp(-0.5) ~ 0.61
            self.param_log_stds.append(head)

        self.to(self.device)

    def forward(self, obs: torch.Tensor) -> Dict[str, Any]:
        x = self.torso(obs)
        logits = self.logits(x)
        value = self.value_head(x).squeeze(-1)
        means = [head(x) for head in self.param_means]
        log_stds = [head(x).clamp(-5.0, 2.0) for head in self.param_log_stds]
        return {"logits": logits, "value": value, "means": means, "log_stds": log_stds}

    @staticmethod
    def _tanh_affine_to_bounds(z: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        # y = low + (tanh(z)+1)/2 * (high-low)
        return low + 0.5 * (torch.tanh(z) + 1.0) * (high - low)

    @staticmethod
    def _atanh_clipped(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # atanh(u) = 0.5 * ln((1+u)/(1-u))
        u = torch.clamp(u, -1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(u) - torch.log1p(-u))

    @staticmethod
    def _log_cosh(z: torch.Tensor) -> torch.Tensor:
        # Numerically stable log(cosh(z))
        # log(cosh(z)) = |z| + log1p(exp(-2|z|)) - log(2)
        t = torch.abs(z)
        return t + torch.log1p(torch.exp(-2.0 * t)) - math.log(2.0)

    def _y_to_z(self, y: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
        # Convert bounded y back to raw z via inverse of tanh-affine
        # u in [-1,1]: y = low + 0.5*(u+1)*(high-low) -> u = 2*(y-low)/(high-low) - 1
        u = 2.0 * (y - low) / (high - low + 1e-8) - 1.0
        z = self._atanh_clipped(u)
        return z

    def _logprob_params(self, z: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor,
                        low: torch.Tensor, high: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z ~ N(mean, std), y = f(z) with f = tanh + affine(low, high)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        log_prob_z = normal.log_prob(z).sum(-1)
        # Change-of-variable correction (stable): sum log |df/dz| = sum [log((high-low)/2) - 2*log_cosh(z)]
        log_det_scale = torch.log(0.5 * (high - low) + 1e-8).sum(-1)
        log_det_tanh = (-2.0 * self._log_cosh(z)).sum(-1)
        log_det = log_det_scale + log_det_tanh
        log_prob_y = log_prob_z - log_det
        # Entropy approximation: base Normal entropy minus E[log|df/dz|] (use current sample; clamp to safe range)
        base_entropy = (0.5 + 0.5 * math.log(2 * math.pi)) * z.shape[-1] + log_std.sum(-1)
        approx_entropy = torch.clamp(base_entropy - log_det, -20.0, 20.0)
        return log_prob_y, approx_entropy

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs: [B, obs_dim]
        out = self.forward(obs)
        logits = out["logits"]
        value = out["value"]
        means = out["means"]
        log_stds = out["log_stds"]

        cat = Categorical(logits=logits)
        aidx = cat.sample()  # [B]

        params_list = []
        lp_params_list = []
        ent_params_list = []
        for i, spec in enumerate(self.param_specs):
            mean_i = means[i]
            log_std_i = log_stds[i]
            # sample raw z then transform
            z_i = mean_i + torch.randn_like(mean_i) * torch.exp(log_std_i)
            low = spec.low.to(self.device)
            high = spec.high.to(self.device)
            y_i = self._tanh_affine_to_bounds(z_i, low, high)
            lp_i, ent_i = self._logprob_params(z_i, mean_i, log_std_i, low, high)
            params_list.append(y_i)
            lp_params_list.append(lp_i)
            ent_params_list.append(ent_i)

        # Select params/logprob/entropy for chosen branch
        batch_idx = torch.arange(obs.shape[0], device=self.device)
        chosen_params = torch.stack([params_list[i][b] for b, i in zip(batch_idx, aidx)], dim=0)
        lp_cont = torch.stack([lp_params_list[i][b] for b, i in zip(batch_idx, aidx)], dim=0)
        ent_cont = torch.stack([ent_params_list[i][b] for b, i in zip(batch_idx, aidx)], dim=0)

        lp_disc = cat.log_prob(aidx)
        logprob = lp_disc + lp_cont
        entropy = cat.entropy() + ent_cont
        return aidx, chosen_params, logprob, value

    def evaluate_actions(self, obs: torch.Tensor, aidx: torch.Tensor, params_padded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute logprob and value for given actions
        out = self.forward(obs)
        logits = out["logits"]
        value = out["value"]
        means = out["means"]
        log_stds = out["log_stds"]

        cat = Categorical(logits=logits)
        lp_disc = cat.log_prob(aidx)
        ent_disc = cat.entropy()

        # Build per-branch logprob for the provided params (padded to max dim)
        lp_cont = torch.zeros_like(lp_disc)
        ent_cont = torch.zeros_like(ent_disc)
        for i, spec in enumerate(self.param_specs):
            mask = (aidx == i)
            if not mask.any():
                continue
            mean_i = means[i][mask]
            log_std_i = log_stds[i][mask]
            low = spec.low.to(self.device)
            high = spec.high.to(self.device)
            # Select corresponding params rows up to the branch dimension
            d = spec.dim
            y_i = params_padded[mask][:, :d]
            # Inverse transform to z
            z_i = self._y_to_z(y_i, low, high)
            lp_i, ent_i = self._logprob_params(z_i, mean_i, log_std_i, low, high)
            lp_cont[mask] = lp_i
            ent_cont[mask] = ent_i

        logprob = lp_disc + lp_cont
        return logprob, ent_disc, ent_cont, value


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, param_specs: List[ParamSpec], device: torch.device):
        self.size = size
        self.obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32, device=device)
        self.aidx_buf = torch.zeros(size, dtype=torch.long, device=device)
        # Params are variable-dim; we will store as a padded tensor with max dim and a mask per action
        self.max_dim = max(s.dim for s in param_specs)
        self.params_buf = torch.zeros((size, self.max_dim), dtype=torch.float32, device=device)
        self.params_mask = torch.zeros((size, self.max_dim), dtype=torch.bool, device=device)
        self.logp_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.val_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.adv_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr = 0
        self.path_start_idx = 0
        self.device = device
        self.param_specs = param_specs

    def store(self, obs, aidx, params, logp, rew, done, val):
        assert self.ptr < self.size
        self.obs_buf[self.ptr] = obs
        self.aidx_buf[self.ptr] = aidx
        # Write params into padded buffer
        spec = self.param_specs[int(aidx.item())]
        d = spec.dim
        self.params_buf[self.ptr, :d] = params[:d]
        self.params_mask[self.ptr, :d] = True
        self.logp_buf[self.ptr] = logp
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val: float, gamma: float, lam: float):
        # GAE-Lambda advantage calculation
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = torch.cat([self.rew_buf[path_slice], torch.tensor([last_val], device=self.device)])
        vals = torch.cat([self.val_buf[path_slice], torch.tensor([last_val], device=self.device)])
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        adv = torch.zeros_like(deltas)
        gae = 0.0
        for t in reversed(range(len(deltas))):
            gae = deltas[t] + gamma * lam * gae * (1.0 - self.done_buf[self.path_start_idx + t])
            adv[t] = gae
        self.adv_buf[path_slice] = adv
        self.ret_buf[path_slice] = adv + self.val_buf[path_slice]
        self.path_start_idx = self.ptr

    def get(self, normalize_adv: bool = True):
        # Allow partial buffers (e.g., when training stops mid-rollout due to total_steps limit)
        n = self.ptr
        assert n > 0, "RolloutBuffer.get called but buffer is empty (ptr == 0)."
        # Slice all tensors to the actually collected size
        obs = self.obs_buf[:n]
        aidx = self.aidx_buf[:n]
        params = self.params_buf[:n]
        params_mask = self.params_mask[:n]
        logp = self.logp_buf[:n]
        ret = self.ret_buf[:n]
        adv = self.adv_buf[:n]
        val = self.val_buf[:n]
        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        data = dict(obs=obs, aidx=aidx, params=params,
                    params_mask=params_mask, logp=logp, ret=ret, adv=adv, val=val)
        return data

    def reset(self):
        self.ptr = 0
        self.path_start_idx = 0


class PPOAgent:
    def __init__(self,
                 obs_dim: int,
                 param_specs: List[ParamSpec],
                 device: torch.device = torch.device('cpu'),
                 hidden_sizes: Tuple[int, int] = (256, 256),
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_coef: float = 0.2,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5):
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.ac = ParamActorCritic(obs_dim, hidden_sizes, param_specs, device).to(device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=lr)

        self.param_specs = param_specs

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> Tuple[int, np.ndarray, float, float]:
        self.ac.eval()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        aidx_t, params_t, logp_t, val_t = self.ac.act(obs_t)
        aidx = int(aidx_t.item())
        params = params_t.squeeze(0).detach().cpu().numpy()
        logp = float(logp_t.item())
        val = float(val_t.item())
        return aidx, params, logp, val

    def update(self, buffer: RolloutBuffer, epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        data = buffer.get()
        obs = data['obs']
        aidx = data['aidx']
        params = data['params']
        mask = data['params_mask']
        old_logp = data['logp']
        ret = data['ret']
        adv = data['adv']
        old_val = data['val']

        n = obs.shape[0]
        idxs = torch.arange(n, device=self.device)
        stats = {}
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                mb_idx = perm[start:start + batch_size]
                obs_b = obs[mb_idx]
                aidx_b = aidx[mb_idx]
                # Extract variable-length parameters per sample according to its branch
                # Here we pass the full padded params; policy will select by aidx mask internally
                params_b = params[mb_idx]
                old_logp_b = old_logp[mb_idx]
                ret_b = ret[mb_idx]
                adv_b = adv[mb_idx]
                old_val_b = old_val[mb_idx]

                # Evaluate current logprob and value using padded params;
                # the network slices to the correct per-branch dimension internally.
                logp, entropy, value = self.ac.evaluate_actions(obs_b, aidx_b, params_b)

                ratio = (logp - old_logp_b).exp()
                approx_kl = (old_logp_b - logp).mean().item()
                clipfracs = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()

                # Policy loss
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_pred = value
                v_pred_clipped = old_val_b + torch.clamp(v_pred - old_val_b, -self.clip_coef, self.clip_coef)
                v_loss_unclipped = (v_pred - ret_b).pow(2)
                v_loss_clipped = (v_pred_clipped - ret_b).pow(2)
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * v_loss + self.ent_coef * ent_loss

                # Finite-loss guard (Fix A): skip update if loss is non-finite
                if not torch.isfinite(loss):
                    # Optional: reduce LR by 50% on a non-finite event
                    for g in self.optimizer.param_groups:
                        g['lr'] = g['lr'] * 0.5
                    # Skip this batch
                    continue

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                stats = {
                    'loss': float(loss.item()),
                    'pg_loss': float(pg_loss.item()),
                    'v_loss': float(v_loss.item()),
                    'entropy': float((-ent_loss).item()),
                    'approx_kl': approx_kl,
                    'clipfrac': clipfracs,
                }
        return stats
