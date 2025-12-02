import argparse
import time
from pathlib import Path
from collections import Counter

import gym
import numpy as np
import torch

import gym_goal  # registers Goal-v0
from gym_goal.wrappers import GoalObsActionWrapper
from gym_goal.agents import PPOAgent
from gym_goal.agents.ppo import ParamActorCritic
from train_ppo import make_param_specs

# Action isimlerini loglarda görmek için
ACTION_NAMES = {
    0: "Kick To",
    1: "Shoot Left",
    2: "Shoot Right",
    3: "Dribble"
}


def safe_torch_load(path: Path, device: torch.device):
    """Load a checkpoint safely across PyTorch versions and devices."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    return ckpt


essd = lambda x: [int(s) for s in x]


def build_agent_from_ckpt(envw: GoalObsActionWrapper, device: torch.device, ckpt: dict) -> PPOAgent:
    param_specs = make_param_specs(envw, device)
    cargs = ckpt.get('args', {})
    hidden_sizes = tuple(cargs.get('hidden_sizes', (256, 256)))
    if not isinstance(hidden_sizes, tuple):
        hidden_sizes = tuple(essd(hidden_sizes))

    agent = PPOAgent(
        obs_dim=int(np.prod(envw.observation_space.shape)),
        param_specs=param_specs,
        device=device,
        hidden_sizes=hidden_sizes,
        lr=cargs.get('lr', 3e-4),
        gamma=cargs.get('gamma', 0.99),
        lam=cargs.get('lam', 0.95),
        clip_coef=cargs.get('clip', 0.2),
        ent_coef=cargs.get('ent_coef', 0.01),
        vf_coef=cargs.get('vf_coef', 0.5),
        max_grad_norm=cargs.get('max_grad_norm', 0.5),
    )
    state = ckpt['ac_state_dict']
    agent.ac.load_state_dict(state)
    agent.ac.eval()
    return agent


def greedy_action(agent: PPOAgent, obs: np.ndarray):
    with torch.no_grad():
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        out = agent.ac.forward(obs_t)
        aidx_t = out["logits"].argmax(dim=-1)
        aidx = int(aidx_t.item())
        mean = out["means"][aidx][0]
        spec = agent.param_specs[aidx]
        params_t = ParamActorCritic._tanh_affine_to_bounds(mean, spec.low.to(agent.device), spec.high.to(agent.device))
        params = params_t.detach().cpu().numpy()
    return aidx, params


def stochastic_action(agent: PPOAgent, obs: np.ndarray):
    aidx, params, _, _ = agent.select_action(obs)
    return aidx, params


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a PPO checkpoint on Goal-v0")
    p.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pt file')
    p.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    p.add_argument('--device', type=str, default='cpu', help='cpu | mps | cuda')
    p.add_argument('--stochastic', action='store_true', help='Use stochastic actions (default: greedy)')
    p.add_argument('--sleep', type=float, default=0.0, help='Seconds to sleep between env frames')
    p.add_argument('--no-render', action='store_true', help='Disable rendering')
    return p.parse_args()


def main():
    args = parse_args()

    dev = args.device.lower()
    if dev == 'mps' and not torch.backends.mps.is_available():
        print('MPS device requested but not available. Falling back to CPU.')
        dev = 'cpu'
    device = torch.device(dev)

    env = gym.make('Goal-v0')
    envw = GoalObsActionWrapper(env)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt_path = Path(args.checkpoint)
    ckpt = safe_torch_load(ckpt_path, device)
    agent = build_agent_from_ckpt(envw, device, ckpt)

    # --- İstatistik Değişkenleri ---
    total_return = 0.0
    successes = 0
    miss_distances = []  # Gol olmayan durumlarda kaleye uzaklık
    episode_steps = []
    action_counts = Counter()  # Hangi aksiyonu kaç kere seçti?

    greedy = not args.stochastic
    mode_str = "Stochastic" if args.stochastic else "Greedy"
    print(f"\nStarting Evaluation ({mode_str} Policy) over {args.episodes} episodes...\n")

    for ep in range(args.episodes):
        obs = envw.reset()
        done = False
        ep_ret = 0.0
        steps = 0

        # Bu epizotta seçilen ilk aksiyonu kaydedelim (stratejiyi anlamak için)
        first_action_logged = False

        while not done:
            if greedy:
                aidx, params = greedy_action(agent, obs)
            else:
                aidx, params = stochastic_action(agent, obs)

            if not first_action_logged:
                action_counts[aidx] += 1
                first_action_logged = True

            obs, reward, done, info = envw.step((aidx, params))
            ep_ret += reward
            steps += 1

            if not args.no_render:
                envw.render()
                if args.sleep > 0:
                    time.sleep(args.sleep)

        total_return += ep_ret
        episode_steps.append(steps)

        # Goal check logic
        is_goal = (ep_ret >= 50.0)
        if is_goal:
            successes += 1
            status = "GOAL! ⚽"
        else:
            status = "Miss ❌"
            # Env reward yapısına göre: gol değilse reward = -distance
            miss_distances.append(-ep_ret)

        print(f"Ep {ep + 1:02d}/{args.episodes} | Return: {ep_ret:6.2f} | Steps: {steps} | {status}")

    # --- Final Raporu ---
    avg_ret = total_return / max(1, args.episodes)
    succ_rate = (successes / max(1, args.episodes)) * 100.0
    avg_steps = sum(episode_steps) / len(episode_steps)
    avg_miss_dist = sum(miss_distances) / len(miss_distances) if miss_distances else 0.0

    print("\n" + "=" * 40)
    print(f"📊 EVALUATION REPORT ({mode_str})")
    print("=" * 40)
    print(f"Total Episodes    : {args.episodes}")
    print(f"Success Rate      : {succ_rate:.1f}%  ({'✅ PASSED' if succ_rate > 90 else ''})")
    print(f"Average Return    : {avg_ret:.2f}")
    print(f"Average Steps     : {avg_steps:.1f}")
    if miss_distances:
        print(f"Avg Miss Distance : {avg_miss_dist:.2f} meters (lower is better)")

    print("-" * 40)
    print("Action Distribution (First move of episode):")
    total_actions = sum(action_counts.values())
    for aidx in sorted(ACTION_NAMES.keys()):
        count = action_counts[aidx]
        pct = (count / total_actions) * 100 if total_actions > 0 else 0
        print(f"  - {ACTION_NAMES[aidx]:<12}: {count:2d} ({pct:.1f}%)")
    print("=" * 40 + "\n")

    env.close()


if __name__ == '__main__':
    main()