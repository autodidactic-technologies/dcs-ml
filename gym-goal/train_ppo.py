import argparse
import time
from pathlib import Path

import gym
import numpy as np
import torch
import wandb

import gym_goal  # registers Goal-v0
from gym_goal.wrappers import GoalObsActionWrapper
from gym_goal.agents import PPOAgent, ParamSpec
from gym_goal.logging_utils import JSONSarsaLogger


def make_param_specs(envw: GoalObsActionWrapper, device: torch.device):
    specs = []
    for ps in envw.param_spaces:
        low = torch.as_tensor(np.array(ps.low, dtype=np.float32).reshape(-1), device=device)
        high = torch.as_tensor(np.array(ps.high, dtype=np.float32).reshape(-1), device=device)
        specs.append(ParamSpec(low=low, high=high))
    return specs


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on gym-goal (parameterized actions)")
    p.add_argument('--total-steps', type=int, default=200_000)
    p.add_argument('--rollout-steps', type=int, default=2048)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--hidden-sizes', type=int, nargs=2, default=[256, 256])
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--vf-coef', type=float, default=0.5)
    p.add_argument('--max-grad-norm', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--save-every', type=int, default=10, help='save checkpoint every N updates')
    p.add_argument('--outdir', type=str, default='runs/ppo_goal')
    # Portability options for checkpoints
    p.add_argument('--save-cpu', action='store_true',
                   help='also save a CPU weights-only checkpoint alongside the main one')
    p.add_argument('--legacy-pt', action='store_true',
                   help='use legacy pickle .pt format (disable zipfile serialization)')
    p.add_argument('--render', action='store_true')
    p.add_argument('--sarsa-json', type=str, default='',
                   help='Path to JSONL file to log SARSA transitions (if empty, disabled)')
    p.add_argument('--resume', type=str, default='', help='Path to a checkpoint to resume training from')
    p.add_argument('--exp-name', type=str, default='exp_01', help='Experiment name for WandB')  # <--- GÜNCELLENDİ
    return p.parse_args()


def main():
    args = parse_args()

    # --- WANDB SETUP (TENSORBOARD YERİNE) ---
    run_name = f"{args.exp_name}_{args.seed}_{int(time.time())}"

    wandb.init(
        project="gym-goal-ppo",  # Senin belirlediğin proje ismi
        name=run_name,  # Run ismi (örn: exp_01_1_17123...)
        config=vars(args),  # Tüm hiperparametreleri otomatik kaydeder
        monitor_gym=False,  # Video kaydını şimdilik kapalı tutuyoruz (hız için)
        save_code=True,  # Kodun o anki halini de buluta yedekler
    )
    # ----------------------------------------

    device = torch.device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make('Goal-v0')
    envw = GoalObsActionWrapper(env)

    obs = envw.reset()
    obs_dim = int(np.prod(envw.observation_space.shape))

    param_specs = make_param_specs(envw, device)
    agent = PPOAgent(obs_dim=obs_dim,
                     param_specs=param_specs,
                     device=device,
                     hidden_sizes=tuple(args.hidden_sizes),
                     lr=args.lr,
                     gamma=args.gamma,
                     lam=args.lam,
                     clip_coef=args.clip,
                     ent_coef=args.ent_coef,
                     vf_coef=args.vf_coef,
                     max_grad_norm=args.max_grad_norm)

    from gym_goal.agents.ppo import RolloutBuffer
    buf = RolloutBuffer(size=args.rollout_steps, obs_dim=obs_dim, param_specs=param_specs, device=device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Optional SARSA JSON logger
    sarsa_logger = JSONSarsaLogger(args.sarsa_json) if args.sarsa_json else None
    if sarsa_logger:
        meta = {
            'args': vars(args),
            'n_actions': int(envw.n_actions),
            'param_dims': [int(np.prod(ps.shape)) for ps in envw.param_spaces],
        }
        sarsa_logger.write_metadata(meta)
    sarsa_pending = None
    episode_idx = 0
    step_in_ep = 0

    ep_ret = 0.0
    ep_len = 0
    global_steps = 0
    update_idx = 0

    if args.resume:
        ckpt_path = Path(args.resume)
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.ac.load_state_dict(ckpt['ac_state_dict'])
        opt_state = ckpt.get('optimizer')
        if opt_state:
            agent.optimizer.load_state_dict(opt_state)
        global_steps = ckpt.get('steps', global_steps)
        update_idx = ckpt.get('update', update_idx)
        print(f"Checkpoint yüklendi: {ckpt_path} | steps={global_steps} update={update_idx}")

    start_t = time.time()
    while global_steps < args.total_steps:
        # Collect rollout
        for _ in range(args.rollout_steps):
            # Sample action for current state
            aidx, params, logp, val = agent.select_action(obs)

            # If we have a pending transition from previous step, finalize it with a_next
            if sarsa_logger and sarsa_pending is not None:
                d_next = param_specs[aidx].dim
                sarsa_logger.write_transition(
                    s=sarsa_pending['s'],
                    a_index=sarsa_pending['aidx'],
                    a_params=sarsa_pending['params'],
                    r=sarsa_pending['r'],
                    s_next=sarsa_pending['s_next'],
                    a_next_index=int(aidx),
                    a_next_params=list(map(float, params[:d_next])),
                    done=sarsa_pending['done'],
                    episode=sarsa_pending['episode'],
                    t_in_ep=sarsa_pending['t'],
                    global_step=sarsa_pending['gstep'],
                )
                sarsa_pending = None

            # Ensure param dims match selected action
            d = param_specs[aidx].dim
            step_action = (aidx, params[:d])
            next_obs, reward, done, info = envw.step(step_action)

            # Store to PPO buffer
            buf.store(torch.as_tensor(obs, dtype=torch.float32, device=device),
                      torch.as_tensor(aidx, dtype=torch.long, device=device),
                      torch.as_tensor(params[:d], dtype=torch.float32, device=device),
                      torch.as_tensor(logp, dtype=torch.float32, device=device),
                      torch.as_tensor(reward, dtype=torch.float32, device=device),
                      torch.as_tensor(done, dtype=torch.float32, device=device),
                      torch.as_tensor(val, dtype=torch.float32, device=device))

            # Prepare SARSA pending
            if sarsa_logger:
                sarsa_pending = {
                    's': list(map(float, np.asarray(obs, dtype=np.float32).tolist())),
                    'aidx': int(aidx),
                    'params': list(map(float, np.asarray(params[:d], dtype=np.float32).tolist())),
                    'r': float(reward),
                    's_next': list(map(float, np.asarray(next_obs, dtype=np.float32).tolist())),
                    'done': bool(done),
                    'episode': int(episode_idx),
                    't': int(step_in_ep),
                    'gstep': int(global_steps + 1),
                }

            ep_ret += reward
            ep_len += 1
            step_in_ep += 1
            global_steps += 1

            obs = next_obs

            if args.render:
                env.render()

            if done:
                # --- LOG EPISODIC RETURN TO WANDB ---
                wandb.log({
                    "charts/episodic_return": ep_ret,
                    "charts/episodic_length": ep_len,
                }, step=global_steps)
                # ------------------------------------

                # Finish SARSA record for terminal step (no a_next)
                if sarsa_logger and sarsa_pending is not None:
                    sarsa_logger.write_transition(
                        s=sarsa_pending['s'],
                        a_index=sarsa_pending['aidx'],
                        a_params=sarsa_pending['params'],
                        r=sarsa_pending['r'],
                        s_next=sarsa_pending['s_next'],
                        a_next_index=None,
                        a_next_params=None,
                        done=True,
                        episode=sarsa_pending['episode'],
                        t_in_ep=sarsa_pending['t'],
                        global_step=sarsa_pending['gstep'],
                    )
                    sarsa_pending = None

                buf.finish_path(last_val=0.0, gamma=args.gamma, lam=args.lam)
                obs = envw.reset()
                print(f"Global Step: {global_steps} | Episode done | return={ep_ret:.2f} length={ep_len}")
                ep_ret, ep_len = 0.0, 0
                step_in_ep = 0
                episode_idx += 1

            if global_steps >= args.total_steps:
                break

        # Time-limit handling
        if buf.path_start_idx < buf.ptr:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                last_val = agent.ac.forward(obs_t)["value"].item()
            buf.finish_path(last_val=last_val, gamma=args.gamma, lam=args.lam)

            if sarsa_logger and sarsa_pending is not None and not sarsa_pending['done']:
                naidx, nparams, _, _ = agent.select_action(obs)
                d_next = param_specs[naidx].dim
                sarsa_logger.write_transition(
                    s=sarsa_pending['s'],
                    a_index=sarsa_pending['aidx'],
                    a_params=sarsa_pending['params'],
                    r=sarsa_pending['r'],
                    s_next=sarsa_pending['s_next'],
                    a_next_index=int(naidx),
                    a_next_params=list(map(float, nparams[:d_next])),
                    done=False,
                    episode=sarsa_pending['episode'],
                    t_in_ep=sarsa_pending['t'],
                    global_step=sarsa_pending['gstep'],
                )
                sarsa_pending = None

        if buf.ptr > 0:
            stats = agent.update(buf, epochs=args.epochs, batch_size=args.batch_size)
            buf.reset()
            update_idx += 1

            # --- LOG TRAINING STATS TO WANDB ---
            fps = int(global_steps / (time.time() - start_t + 1e-8))
            wandb.log({
                "losses/total_loss": stats.get('loss', 0),
                "losses/value_loss": stats.get('v_loss', 0),
                "losses/policy_loss": stats.get('pg_loss', 0),
                "losses/entropy": stats.get('entropy', 0),
                "losses/approx_kl": stats.get('approx_kl', 0),
                "charts/SPS": fps,
            }, step=global_steps)
            # -----------------------------------

            if update_idx % args.save_every == 0:
                ckpt_path = outdir / f'checkpoint_{update_idx:05d}.pt'
                ckpt = {
                    'ac_state_dict': agent.ac.state_dict(),
                    'optimizer': agent.optimizer.state_dict(),
                    'args': vars(args),
                    'update': update_idx,
                    'steps': global_steps,
                }
                torch.save(
                    ckpt,
                    ckpt_path,
                    _use_new_zipfile_serialization=not args.legacy_pt,
                )
                print(f"Checkpoint kaydedildi: {ckpt_path}")
                if args.save_cpu:
                    cpu_state = {k: v.detach().to('cpu') for k, v in agent.ac.state_dict().items()}
                    cpu_ckpt = {
                        'ac_state_dict': cpu_state,
                        'args': vars(args),
                        'update': update_idx,
                        'steps': global_steps,
                    }
                    torch.save(
                        cpu_ckpt,
                        outdir / f'checkpoint_{update_idx:05d}.cpu.pt',
                        _use_new_zipfile_serialization=not args.legacy_pt,
                    )

            if (update_idx % 1) == 0:
                print(
                    f"Update {update_idx} | steps {global_steps}/{args.total_steps} | loss {stats.get('loss', 0):.3f} | "
                    f"fps {fps}")
        else:
            break

    env.close()
    wandb.finish()  # <--- EKLENDİ


if __name__ == '__main__':
    main()