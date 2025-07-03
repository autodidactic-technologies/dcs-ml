"""
ENHANCED Training System - Added PPO Baseline
Environment ve Mediator reward'ları uyumlu hale getirdik
PPO-only baseline eklendi karşılaştırma için
"""

import os
import torch
import numpy as np
from loguru import logger
from typing import Dict
import re
import wandb

# Sizin sistemin importları
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda
from stable_baselines3 import PPO

# Sizin dosyalarınızdan importlar
from TSCAssistant.tsc_assistant_mediator import TSCAgentWithMediator
from utils.make_tsc_env import make_env


def get_device():
    """Device detection"""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class Config:
    """Clean config"""

    def __init__(self):
        # Environment
        self.env_name = "MiniGrid-DoorKey-6x6-v0"
        self.max_steps = 100

        # Training
        self.max_episodes = 3000
        self.exploration_episodes = 2000

        # LLM
        self.llm_model = "llama3.1:8b"
        self.llm_temperature = 0.1

        # Logging
        self.log_every = 10
        self.save_every = 25

        # WandB
        self.use_wandb = True
        self.wandb_entity = "BILGEM_DCS_RL"
        self.wandb_project = "mediator-fixed-rewards"


def setup_wandb(config: Config, experiment_type: str = "mediator"):
    """Setup WandB"""
    if not config.use_wandb:
        return None

    try:
        project_name = f"{config.wandb_project}-{experiment_type}"
        wandb.init(
            entity=config.wandb_entity,
            project=project_name,
            config={
                "episodes": config.max_episodes,
                "env": config.env_name,
                "llm": config.llm_model,
                "system": f"fixed_rewards_v1_{experiment_type}",
                "experiment_type": experiment_type
            }
        )
        print(f"✅ WandB ready for {experiment_type}")
        return wandb
    except Exception as e:
        print(f"⚠️ WandB failed: {e}")
        config.use_wandb = False
        return None


def get_phase_info(episode: int, config: Config) -> Dict:
    """
    NEW: Get comprehensive phase information
    """
    exploration_episodes = config.exploration_episodes
    max_episodes = config.max_episodes

    if episode < exploration_episodes:
        phase = "EXPLORATION"
        progress = (episode / exploration_episodes) * 100
        remaining = exploration_episodes - episode
        phase_total = exploration_episodes
        phase_current = episode
    else:
        phase = "EXPLOITATION"
        exploit_episode = episode - exploration_episodes
        max_exploit = max_episodes - exploration_episodes
        progress = (exploit_episode / max_exploit) * 100 if max_exploit > 0 else 100
        remaining = max_episodes - episode
        phase_total = max_exploit
        phase_current = exploit_episode

    return {
        'phase': phase,
        'progress': progress,
        'remaining': remaining,
        'phase_current': phase_current,
        'phase_total': phase_total,
        'is_transition': episode == exploration_episodes
    }


def log_enhanced_progress(episode: int, result: Dict, config: Config, results: list):
    """
    NEW: Enhanced progress logging with phase information
    """
    if episode % config.log_every == 0:
        phase_info = get_phase_info(episode, config)
        recent_results = results[-10:] if len(results) >= 10 else results

        # Calculate metrics
        avg_env_reward = np.mean([r['env_reward'] for r in recent_results])
        success_rate = np.mean([r['success'] for r in recent_results])
        avg_interrupts = np.mean([r['interrupts'] for r in recent_results])
        avg_efficiency = np.mean([r['efficiency'] for r in recent_results])

        # Phase-specific metrics
        phase_results = [r for r in results if r['phase'] == phase_info['phase']]
        if phase_results:
            phase_success = np.mean([r['success'] for r in phase_results])
            phase_reward = np.mean([r['env_reward'] for r in phase_results])
        else:
            phase_success = 0
            phase_reward = 0

        # Create progress bar
        bar_length = 20
        filled_length = int(bar_length * phase_info['progress'] / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        print(f"\n📊 EPISODE {episode:4d} [{phase_info['phase']:11s}] Progress: {phase_info['progress']:5.1f}% |{bar}|")
        print(
            f"   Recent (10): Success={success_rate:.1%}, Reward={avg_env_reward:5.2f}, LLM={avg_interrupts:.1f}, Eff={avg_efficiency:.1%}")
        print(
            f"   Phase Stats: Success={phase_success:.1%}, Reward={phase_reward:5.2f} ({phase_info['phase_current']}/{phase_info['phase_total']})")
        print(f"   Remaining: {phase_info['remaining']} episodes")

        # Phase transition warning
        if phase_info['remaining'] <= 10 and phase_info['phase'] == "EXPLORATION":
            print("   ⚠️  APPROACHING EXPLOITATION PHASE!")
        elif phase_info['remaining'] <= 10 and phase_info['phase'] == "EXPLOITATION":
            print("   🏁 TRAINING ALMOST COMPLETE!")


def calculate_aligned_mediator_reward(env_reward: float, was_interrupted: bool,
                                      plan_changed: bool, step: int, episode: int,
                                      recent_performance: float = 0.5, config: Config = None) -> float:
    """
    SMART: Dynamic mediator reward calculation

    Factors:
    - Environment reward magnitude and direction
    - Asking behavior effectiveness
    - Episode progress (early vs late)
    - Recent performance trend
    - Step efficiency
    """

    # 1. BASE REWARD: Scaled environment reward
    base_reward = env_reward * 0.8  # Scale down to leave room for modifiers

    # 2. ASKING BEHAVIOR ANALYSIS
    asking_modifier = 0.0

    if was_interrupted:
        if plan_changed:
            # Plan değiştirdi - effectiveness depends on outcome
            if env_reward > 0:
                # Good override -> outcome positive
                asking_modifier = 0.3 + (env_reward * 0.2)  # Scale with success
            elif env_reward == 0:
                # Neutral override -> small positive (tried to help)
                asking_modifier = 0.1
            else:
                # Bad override -> outcome negative, but maybe needed
                asking_modifier = -0.1 + (env_reward * 0.1)  # Less penalty if env_reward not too bad
        else:
            # Plan değiştirmedi - wasted LLM call
            # Penalty scales with how bad the waste was
            if env_reward >= 0:
                # Environment is fine, asking was unnecessary
                asking_modifier = -0.15 - (0.05 * step / 50)  # More penalty in later steps
            else:
                # Environment struggling, asking was reasonable but ineffective
                asking_modifier = -0.08
    else:
        # Didn't ask - evaluate if this was smart
        if env_reward > 0:
            # Good outcome without asking - efficiency bonus
            efficiency_bonus = 0.1 + (env_reward * 0.05)
            # Scale by episode progress (more bonus when experienced)
            episode_scale = min(1.5, 1.0 + (episode / 100))
            asking_modifier = efficiency_bonus * episode_scale
        elif env_reward < -0.05:
            # Bad outcome, maybe should have asked
            # Penalty depends on how bad and recent performance
            should_have_asked_penalty = abs(env_reward) * 0.3
            # If recent performance is bad, bigger penalty for not asking
            performance_scale = 2.0 - recent_performance  # 1.5 if perf=0.5, 2.0 if perf=0
            asking_modifier = -should_have_asked_penalty * performance_scale
        else:
            # Neutral step, no asking needed
            asking_modifier = 0.02  # Small efficiency bonus

    # 3. STEP EFFICIENCY: Penalize long episodes
    step_efficiency = 0.0
    if step > 60:
        # Progressive penalty for long episodes
        step_efficiency = -0.002 * (step - 60) ** 1.2
    elif step < 30 and env_reward > 0:
        # Bonus for quick success
        step_efficiency = 0.05 * (30 - step) / 30

    # 4. EPISODE PROGRESS: Learning phase adjustment (FIXED)
    learning_adjustment = 0.0
    exploration_episodes = config.exploration_episodes if config else 50

    if episode < exploration_episodes * 0.5:  # Early exploration (first half)
        # Be more forgiving of exploration and asking behavior
        if asking_modifier < 0:
            learning_adjustment = abs(asking_modifier) * 0.4  # Reduce penalties more
        if was_interrupted:
            learning_adjustment += 0.05  # Small bonus for exploration asking
    elif episode < exploration_episodes:  # Late exploration
        # Start to prefer more efficiency but still allow learning
        if asking_modifier < 0:
            learning_adjustment = abs(asking_modifier) * 0.2  # Reduce penalties less
    else:  # Exploitation phase
        # Expect efficiency and good decisions
        if asking_modifier > 0 and not was_interrupted:
            learning_adjustment = asking_modifier * 0.3  # Bonus for efficiency
        elif was_interrupted and not plan_changed:
            learning_adjustment = -0.05  # Penalty for unnecessary interrupts in exploitation

    # 5. COMBINE ALL FACTORS
    final_reward = base_reward + asking_modifier + step_efficiency + learning_adjustment

    # 6. REASONABLE BOUNDS: Keep reward in reasonable range
    final_reward = max(-1.0, min(2.0, final_reward))

    return final_reward


def run_ppo_only_episode(rl_agent, rl_env, episode: int, max_steps: int = 100):
    """
    NEW: PPO-only baseline episode
    Sadece PPO ajanı çalışır, LLM kullanılmaz
    """
    # Reset environment
    obs, _ = rl_env.reset(seed=episode)

    done = False
    step = 0
    total_reward = 0

    logger.info(f"PPO-Only Episode {episode} START")

    while not done and step < max_steps:
        step += 1

        # Only RL agent decision
        if rl_agent:
            action, _ = rl_agent.predict(obs, deterministic=True)
            action = int(action)
        else:
            action = rl_env.action_space.sample()

        # Environment step
        try:
            obs, reward, terminated, truncated, _ = rl_env.step(action)
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            reward = -0.1  # Penalty for failure
            obs, _, terminated, truncated, _ = rl_env.step(0)

        done = terminated or truncated
        total_reward += reward

    # Episode end
    episode_success = total_reward > 0

    logger.info(f"PPO-Only Episode {episode} END: "
                f"Reward={total_reward:.2f}, "
                f"Steps={step}, "
                f"Success={episode_success}")

    return {
        'env_reward': total_reward,
        'mediator_reward': 0.0,  # No mediator in PPO-only
        'reward_alignment': 1.0,  # Perfect alignment (no mediator)
        'steps': step,
        'success': episode_success,
        'interrupts': 0,  # No LLM calls
        'overrides': 0,  # No LLM calls
        'efficiency': 1.0,  # No LLM calls = perfect efficiency
        'phase': "PPO_ONLY"
    }


def run_episode_flow(rl_agent, tsc_agent, rl_env, llm_env, episode: int, max_steps: int = 100, results: list = None,
                     config: Config = None):
    """
    FIXED episode flow - Smart reward calculation with proper phase tracking
    """
    # Reset environments
    rl_obs, _ = rl_env.reset(seed=episode)
    llm_obs, llm_info = llm_env.reset(seed=episode)

    done = False
    sim_step = 0
    total_env_reward = 0
    total_mediator_reward = 0
    interrupts = 0
    overrides = 0

    llm_info["llm_env"] = llm_env

    # FIXED: Phase tracking based on config
    if config:
        exploration_episodes = config.exploration_episodes
    else:
        exploration_episodes = 50  # Fallback only if config is None

    phase = "EXPLORATION" if episode < exploration_episodes else "EXPLOITATION"

    # Calculate progress percentages
    if config:
        if episode < exploration_episodes:
            progress = (episode / exploration_episodes) * 100
            remaining = exploration_episodes - episode
            logger.info(
                f"Episode {episode} START [Phase: {phase}] - Progress: {progress:.1f}% ({remaining} episodes to exploitation)")
        else:
            exploit_episode = episode - exploration_episodes
            max_exploit = config.max_episodes - exploration_episodes
            progress = (exploit_episode / max_exploit) * 100 if max_exploit > 0 else 100
            remaining = config.max_episodes - episode
            logger.info(
                f"Episode {episode} START [Phase: {phase}] - Progress: {progress:.1f}% ({remaining} episodes remaining)")

        # Phase transition detection
        if episode == exploration_episodes:
            logger.info("🔄 PHASE TRANSITION: Switching from EXPLORATION to EXPLOITATION")
    else:
        logger.info(f"Episode {episode} START [Phase: {phase}] - No config provided")

    while not done and sim_step < max_steps:
        sim_step += 1

        # RL AGENT DECISION
        if rl_agent:
            rl_action, _ = rl_agent.predict(rl_obs, deterministic=True)
            rl_action = int(rl_action)
        else:
            rl_action = rl_env.action_space.sample()

        # TSC AGENT DECISION
        final_action, was_interrupted, interaction_info = tsc_agent.agent_run(
            sim_step=sim_step,
            obs=llm_obs,
            rl_action=rl_action,
            infos=llm_info,
            reward=total_env_reward,
            use_learned_asking=True
        )

        # STATS
        if was_interrupted:
            interrupts += 1
            if interaction_info.get('llm_plan_changed', False):
                overrides += 1

        # ENVIRONMENT STEP
        try:
            rl_obs, env_reward, terminated, truncated, _ = rl_env.step(final_action)
            llm_obs, _, _, _, llm_info = llm_env.step(final_action)
            llm_info["llm_env"] = llm_env
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            env_reward = -0.1  # Penalty for failure
            rl_obs, _, terminated, truncated, _ = rl_env.step(0)
            llm_obs, _, _, _, llm_info = llm_env.step(0)
            llm_info["llm_env"] = llm_env

        done = terminated or truncated
        total_env_reward += env_reward

        # ALIGNED MEDIATOR TRAINING
        if sim_step > 1:
            # Get recent performance for smart reward calculation
            recent_performance = 0.5  # Default
            if len(results) >= 5:
                recent_performance = np.mean([r['success'] for r in results[-5:]])

            # SMART: Dynamic reward calculation (with config)
            mediator_reward = calculate_aligned_mediator_reward(
                env_reward=env_reward,
                was_interrupted=was_interrupted,
                plan_changed=interaction_info.get('llm_plan_changed', False),
                step=sim_step,
                episode=episode,
                recent_performance=recent_performance,
                config=config  # Pass config for phase info
            )

            total_mediator_reward += mediator_reward

            # Train mediator with smart reward
            tsc_agent.mediator.train_asking_policy(
                obs=llm_obs,
                action=rl_action,
                reward=mediator_reward,  # SMART reward
                next_obs=llm_obs,
                asked_llm=was_interrupted,
                llm_plan_changed=interaction_info.get('llm_plan_changed', False)
            )

    # Episode end
    episode_success = total_env_reward > 0
    tsc_agent.update_performance(episode_success)
    tsc_agent.mediator.episode_end(episode_success)

    # Efficiency
    efficiency = overrides / max(interrupts, 1) if interrupts > 0 else 1.0

    logger.info(f"Episode {episode} [{phase}] END: "
                f"Env Reward={total_env_reward:.2f}, "
                f"Mediator Reward={total_mediator_reward:.2f}, "
                f"Interrupts={interrupts}, Efficiency={efficiency:.1%}")

    return {
        'env_reward': total_env_reward,
        'mediator_reward': total_mediator_reward,
        'reward_alignment': total_mediator_reward / max(abs(total_env_reward), 0.1),  # Alignment metric
        'steps': sim_step,
        'success': episode_success,
        'interrupts': interrupts,
        'overrides': overrides,
        'efficiency': efficiency,
        'phase': phase
    }


def log_to_wandb(episode: int, result: Dict, config: Config):
    """Enhanced WandB logging with alignment metrics"""
    if not config.use_wandb:
        return

    try:
        wandb.log({
            "episode": episode,
            "env_reward": result['env_reward'],
            "mediator_reward": result['mediator_reward'],
            "reward_alignment": result['reward_alignment'],  # NEW: Alignment metric
            "success": int(result['success']),
            "efficiency": result['efficiency'],
            "interrupts": result['interrupts'],
            "overrides": result['overrides'],
            "phase": result['phase']
        })
    except Exception as e:
        pass


def check_available_envs():
    """Check available environments"""
    import gymnasium as gym
    try:
        env_candidates = ["MiniGrid-DoorKey-6x6-v0"]
        for env_name in env_candidates:
            try:
                env = gym.make(env_name)
                env.close()
                print(f"✅ {env_name} available")
                return env_name
            except:
                print(f"❌ {env_name} not found")
        return None
    except:
        return None


def run_comparison_experiment(config: Config, rl_agent, tsc_agent, rl_env, llm_env):
    """
    NEW: Run both PPO-only and Mediator experiments for comparison
    """
    print("\n🔬 COMPARISON EXPERIMENT")
    print("=" * 50)

    # 1. PPO-ONLY BASELINE
    print("🤖 Running PPO-Only Baseline...")
    wandb_ppo = setup_wandb(config, "ppo-only")

    ppo_results = []
    for episode in range(config.max_episodes):
        try:
            result = run_ppo_only_episode(
                rl_agent=rl_agent,
                rl_env=rl_env,
                episode=episode
            )
            ppo_results.append(result)

            # Log to WandB
            if config.use_wandb and wandb_ppo:
                wandb_ppo.log({
                    "episode": episode,
                    "env_reward": result['env_reward'],
                    "success": int(result['success']),
                    "steps": result['steps'],
                    "experiment": "ppo_only"
                })

        except Exception as e:
            logger.error(f"PPO Episode {episode} failed: {e}")
            continue

    if wandb_ppo:
        wandb_ppo.finish()

    # PPO Results
    ppo_success_rate = np.mean([r['success'] for r in ppo_results])
    ppo_avg_reward = np.mean([r['env_reward'] for r in ppo_results])
    ppo_avg_steps = np.mean([r['steps'] for r in ppo_results])

    print(f"\n📊 PPO-ONLY RESULTS:")
    print(f"Success Rate: {ppo_success_rate:.1%}")
    print(f"Avg Reward: {ppo_avg_reward:.3f}")
    print(f"Avg Steps: {ppo_avg_steps:.1f}")
    print(f"LLM Interactions: 0 (Pure RL)")

    # 2. MEDIATOR SYSTEM
    print(f"\n🧠 Running Mediator System...")
    wandb_mediator = setup_wandb(config, "mediator")

    mediator_results = []
    for episode in range(config.max_episodes):
        try:
            result = run_episode_flow(
                rl_agent=rl_agent,
                tsc_agent=tsc_agent,
                rl_env=rl_env,
                llm_env=llm_env,
                episode=episode,
                results=mediator_results,
                config=config
            )
            mediator_results.append(result)

            # Log to WandB
            log_to_wandb(episode, result, config)

        except Exception as e:
            logger.error(f"Mediator Episode {episode} failed: {e}")
            continue

    if wandb_mediator:
        wandb_mediator.finish()

    # Mediator Results
    med_success_rate = np.mean([r['success'] for r in mediator_results])
    med_avg_reward = np.mean([r['env_reward'] for r in mediator_results])
    med_avg_steps = np.mean([r['steps'] for r in mediator_results])
    med_avg_interrupts = np.mean([r['interrupts'] for r in mediator_results])
    med_avg_efficiency = np.mean([r['efficiency'] for r in mediator_results])

    print(f"\n📊 MEDIATOR RESULTS:")
    print(f"Success Rate: {med_success_rate:.1%}")
    print(f"Avg Reward: {med_avg_reward:.3f}")
    print(f"Avg Steps: {med_avg_steps:.1f}")
    print(f"Avg LLM Interactions: {med_avg_interrupts:.1f}")
    print(f"Avg Efficiency: {med_avg_efficiency:.1%}")

    # 3. COMPARISON
    print(f"\n🆚 COMPARISON:")
    print(f"{'Metric':<20} {'PPO-Only':<12} {'Mediator':<12} {'Improvement':<15}")
    print("-" * 60)

    success_improvement = med_success_rate - ppo_success_rate
    reward_improvement = med_avg_reward - ppo_avg_reward
    step_comparison = med_avg_steps - ppo_avg_steps

    print(f"{'Success Rate':<20} {ppo_success_rate:<12.1%} {med_success_rate:<12.1%} {success_improvement:+.1%}")
    print(f"{'Avg Reward':<20} {ppo_avg_reward:<12.3f} {med_avg_reward:<12.3f} {reward_improvement:+.3f}")
    print(f"{'Avg Steps':<20} {ppo_avg_steps:<12.1f} {med_avg_steps:<12.1f} {step_comparison:+.1f}")
    print(f"{'LLM Calls':<20} {'0':<12} {med_avg_interrupts:<12.1f} {'+' + str(med_avg_interrupts)}")

    # Assessment
    print(f"\n🎯 ASSESSMENT:")
    if med_success_rate > ppo_success_rate + 0.05:  # 5% improvement threshold
        print("✅ MEDIATOR WINS: Significantly better success rate!")
        if med_avg_interrupts < 10:
            print("🎉 EXCELLENT: High improvement with few LLM calls!")
        else:
            print("⚠️ COSTLY: Good results but many LLM calls")
    elif abs(med_success_rate - ppo_success_rate) < 0.05:  # Similar performance
        print("🤝 TIE: Similar performance")
        if med_avg_interrupts > 5:
            print("❌ PPO PREFERRED: Same results, but mediator uses LLM unnecessarily")
        else:
            print("✅ MEDIATOR ACCEPTABLE: Similar results with minimal LLM usage")
    else:
        print("❌ PPO WINS: Pure RL performs better")

    return {
        'ppo_results': ppo_results,
        'mediator_results': mediator_results,
        'comparison': {
            'ppo_success': ppo_success_rate,
            'mediator_success': med_success_rate,
            'ppo_reward': ppo_avg_reward,
            'mediator_reward': med_avg_reward,
            'mediator_llm_calls': med_avg_interrupts
        }
    }


def main():
    """Main function with ENHANCED system including PPO baseline"""

    print("🔧MEDIATOR TRAINING SYSTEM")
    print("✅ Environment-Mediator reward alignment")
    print("✅ Proper exploration-exploitation")
    print("✅ Clean WandB integration")
    print("🆚 PPO-only baseline for comparison")
    print("=" * 50)

    # Setup
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Check requirements
    try:
        import gymnasium as gym
        import minigrid
        print("✅ Gymnasium & MiniGrid ready")
    except ImportError:
        print("❌ Missing packages")
        return

    # Environment detection
    available_env = check_available_envs()
    if not available_env:
        print("❌ No MiniGrid environment found!")
        return

    config = Config()
    config.env_name = available_env
    print(f"🎯 Environment: {config.env_name}")

    # Device
    device = get_device()
    print(f"🚀 Device: {device}")

    # Initialize LLM
    try:
        chat = ChatOllama(
            model=config.llm_model,
            temperature=config.llm_temperature,
            num_predict=200
        )
        llm = RunnableLambda(lambda x: chat.invoke(x))
        print(f"✅ LLM initialized: {config.llm_model}")

        # Test connection
        test_response = llm.invoke("Hello, respond with just 'OK'")
        print(f"✅ LLM test successful")
    except Exception as e:
        print(f"❌ LLM failed: {e}")
        return

    # Initialize environments
    try:
        rl_env, llm_env = make_env(env_name=config.env_name, max_steps=config.max_steps)
        print(f"✅ Environments created")
        obs_shape = llm_env.observation_space['image'].shape
        print(f"📐 Obs shape: {obs_shape}")
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        return

    # Load RL Agent
    rl_agent = None
    rl_paths = ["models/ppo_minigrid_doorkey_6x6.zip"]

    for path in rl_paths:
        try:
            rl_agent = PPO.load(path, device=device)
            print(f"✅ RL agent loaded: {path}")
            break
        except:
            continue

    if not rl_agent:
        print("⚠️ RL agent not found, using random actions")

    # Initialize TSC Agent
    tsc_agent = TSCAgentWithMediator(
        llm=llm,
        obs_shape=obs_shape,
        device=str(device),
        verbose=True,
        train_mediator=True
    )
    print("✅ TSC Agent initialized")

    # Check for existing trained models
    existing_models = []
    model_paths = [
        "checkpoints/mediator_final_3000ep.pt",
        "checkpoints/mediator_final_quick_test.pt"
    ]

    for path in model_paths:
        if os.path.exists(path):
            existing_models.append(path)

    if existing_models:
        print(f"💾 Found {len(existing_models)} existing trained model(s):")
        for model in existing_models:
            print(f"   - {model}")
        print("💡 You can use these for evaluation and comparison")

    # MAIN LOOP - Ana menüye geri dön
    while True:
        # Training menu
        print("\n📋 ENHANCED TRAINING OPTIONS:")
        print("1. Quick Test (20 episodes)")
        print("2. Full Training (3000 episodes)")
        print("3. PPO-Only Baseline (no LLM)")
        print("4. Comparison Experiment (PPO vs Mediator)")
        print("5. Evaluation Only")
        print("6. Exit")

        choice = input("\nChoice (1-6): ").strip()

        if choice == "1":
            config.max_episodes = 20
            config.exploration_episodes = 10
            print("🚀 Quick test starting...")

            # Setup WandB
            wandb_instance = setup_wandb(config, "quick-test")

            # Training loop for mediator
            results = []
            for episode in range(config.max_episodes):
                try:
                    result = run_episode_flow(
                        rl_agent=rl_agent,
                        tsc_agent=tsc_agent,
                        rl_env=rl_env,
                        llm_env=llm_env,
                        episode=episode,
                        results=results,
                        config=config  # Pass config for phase tracking
                    )
                    results.append(result)
                    log_to_wandb(episode, result, config)

                    # Enhanced progress logging
                    log_enhanced_progress(episode, result, config, results)
                except Exception as e:
                    logger.error(f"Episode {episode} failed: {e}")
                    continue

            if config.use_wandb and wandb_instance:
                wandb_instance.finish()

            print("✅ Quick test completed!")
            input("\nPress Enter to return to main menu...")

        elif choice == "2":
            print("🚀 Full training starting...")

            # Setup WandB
            wandb_instance = setup_wandb(config, "full-training")

            # Training loop for mediator
            results = []
            for episode in range(config.max_episodes):
                try:
                    result = run_episode_flow(
                        rl_agent=rl_agent,
                        tsc_agent=tsc_agent,
                        rl_env=rl_env,
                        llm_env=llm_env,
                        episode=episode,
                        results=results,
                        config = config  # Pass config for phase tracking
                    )
                    results.append(result)
                    log_to_wandb(episode, result, config)

                    # Progress log
                    if episode % config.log_every == 0:
                        recent_results = results[-10:] if len(results) >= 10 else results
                        avg_env_reward = np.mean([r['env_reward'] for r in recent_results])
                        success_rate = np.mean([r['success'] for r in recent_results])
                        print(f"Episode {episode:3d} | Env: {avg_env_reward:5.2f} | Success: {success_rate:.1%}")

                except Exception as e:
                    logger.error(f"Episode {episode} failed: {e}")
                    continue

            if config.use_wandb and wandb_instance:
                wandb_instance.finish()

            print("✅ Full training completed!")
            input("\nPress Enter to return to main menu...")

        elif choice == "3":
            print("🤖 PPO-Only Baseline starting...")
            original_wandb = config.use_wandb
            config.use_wandb = False

            # Run PPO-only baseline
            results = []
            for episode in range(config.max_episodes):
                result = run_ppo_only_episode(rl_agent, rl_env, episode)
                results.append(result)

            # Results
            success_rate = np.mean([r['success'] for r in results])
            avg_reward = np.mean([r['env_reward'] for r in results])
            avg_steps = np.mean([r['steps'] for r in results])

            print(f"\n📊 PPO-ONLY BASELINE RESULTS:")
            print(f"Success Rate: {success_rate:.1%}")
            print(f"Avg Reward: {avg_reward:.3f}")
            print(f"Avg Steps: {avg_steps:.1f}")
            print(f"LLM Interactions: 0 (Pure RL)")

            # Restore original wandb setting
            config.use_wandb = original_wandb

            print("✅ PPO baseline completed!")
            input("\nPress Enter to return to main menu...")

        elif choice == "4":
            print("🆚 Comparison Experiment starting...")
            original_episodes = config.max_episodes
            config.max_episodes = min(config.max_episodes, 50)  # Shorter for comparison

            comparison_results = run_comparison_experiment(config, rl_agent, tsc_agent, rl_env, llm_env)
            print("✅ Comparison experiment completed!")

            # Restore original episodes
            config.max_episodes = original_episodes

            input("\nPress Enter to return to main menu...")

        elif choice == "5":
            original_wandb = config.use_wandb
            config.use_wandb = False
            print("🔍 Evaluation only...")
            # Run evaluation
            results = []
            for episode in range(20):
                result = run_episode_flow(rl_agent, tsc_agent, rl_env, llm_env, episode, config=config)
                results.append(result)

            # Results
            avg_env_reward = np.mean([r['env_reward'] for r in results])
            success_rate = np.mean([r['success'] for r in results])
            avg_alignment = np.mean([r['reward_alignment'] for r in results])

            print(f"\n📊 EVALUATION RESULTS:")
            print(f"Success Rate: {success_rate:.1%}")
            print(f"Avg Env Reward: {avg_env_reward:.3f}")
            print(f"Reward Alignment: {avg_alignment:.3f}")

            # Restore original wandb setting
            config.use_wandb = original_wandb

            print("✅ Evaluation completed!")
            input("\nPress Enter to return to main menu...")

        elif choice == "6":
            print("👋 Exit")
            break
        else:
            print("❌ Invalid choice")
            input("\nPress Enter to try again...")

    # Cleanup
    rl_env.close()
    llm_env.close()

    print("\n✨ Enhanced training system completed!")


if __name__ == "__main__":
    main()