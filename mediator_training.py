import os
import torch
import numpy as np
import wandb
from loguru import logger
from typing import Dict

# Llama support
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableLambda

# Import the TSC assistant - make sure this points to your file
from TSCAssistant.tsc_assistant_mediator import TSCAgentWithMediator
from utils.make_tsc_env import make_env
from stable_baselines3 import PPO


def get_device():
    """Get the best available device."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.backends.cuda.is_built() and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def run_episode_flow(rl_agent, tsc_agent, rl_env, llm_env, episode: int, max_steps: int = 100):
    """
    Enhanced episode flow with learning-aware monitoring and adaptive feedback
    """
    # Reset environments
    rl_obs, _ = rl_env.reset(seed=episode)
    llm_obs, llm_info = llm_env.reset(seed=episode)

    done = False
    sim_step = 0
    total_reward = 0
    interrupts = 0
    overrides = 0
    agreements = 0
    step_rewards = []

    episode_success = False
    steps_without_progress = 0
    last_reward = 0

    # Learning phase tracking
    learning_phase = tsc_agent.mediator.learning_phase
    learning_bonus_applied = 0

    llm_info["llm_env"] = llm_env
    logger.info(f"Episode {episode} START [Phase: {learning_phase.upper()}] - RL is primary agent")

    while not done and sim_step < max_steps:
        sim_step += 1

        # RL AGENT MAKES PRIMARY DECISION
        rl_action, _ = rl_agent.predict(rl_obs, deterministic=True)
        rl_action = int(rl_action)

        # MEDIATOR DECIDES (with learning awareness)
        final_action, was_interrupted, interaction_info = tsc_agent.agent_run(
            sim_step=sim_step,
            obs=llm_obs,
            rl_action=rl_action,
            infos={"env": "MiniGrid-DoorKey-6x6-v0", "llm_env": llm_env},
            reward=total_reward,
            use_learned_asking=True
        )

        # Track learning-aware metrics
        if was_interrupted:
            interrupts += 1
            if interaction_info.get('llm_plan_changed', False):
                overrides += 1
            else:
                agreements += 1

        # EXECUTE ACTION IN ENVIRONMENT
        try:
            rl_obs, reward, terminated, truncated, _ = rl_env.step(final_action)
            llm_obs, _, _, _, llm_info = llm_env.step(final_action)
            llm_info["llm_env"] = llm_env
        except Exception as e:
            logger.error(f"Environment step failed: {e}")
            rl_obs, reward, terminated, truncated, _ = rl_env.step(0)
            llm_obs, _, _, _, llm_info = llm_env.step(0)
            llm_info["llm_env"] = llm_env

        done = terminated or truncated
        total_reward += reward
        step_rewards.append(reward)

        if reward > last_reward:
            steps_without_progress = 0
        else:
            steps_without_progress += 1
        last_reward = total_reward

        if total_reward > 0:
            episode_success = True

        # LEARNING-AWARE mediator training
        if sim_step > 1:
            # Base mediator reward
            mediator_reward = reward

            # Learning phase adjustments
            if learning_phase == "early_exploration":
                # More forgiving during early learning
                if was_interrupted and interaction_info.get('llm_plan_changed', False):
                    learning_bonus = 0.1  # Bonus for learning to interrupt
                    mediator_reward += learning_bonus
                    learning_bonus_applied += learning_bonus
                elif was_interrupted and not interaction_info.get('llm_plan_changed', False):
                    # Reduced penalty for agreements during learning
                    agreement_penalty = 0.03
                    mediator_reward -= agreement_penalty

            elif learning_phase == "guided_learning":
                # Moderate adjustments during guided learning
                if agreements > interrupts * 0.6:  # If agreeing too much
                    efficiency_penalty = 0.05
                    mediator_reward -= efficiency_penalty
                elif episode_success and sim_step < max_steps * 0.7:
                    success_bonus = 0.08
                    mediator_reward += success_bonus
                    learning_bonus_applied += success_bonus

            else:  # autonomous phase
                # Standard efficiency-based rewards
                if agreements > 0:
                    agreement_penalty = (agreements / max(interrupts, 1)) * 0.15
                    mediator_reward -= agreement_penalty

            # Success bonus (scaled by learning phase)
            success_multiplier = 1.2 if learning_phase == "early_exploration" else 1.0
            if episode_success and sim_step < max_steps * 0.8:
                success_bonus = 0.1 * success_multiplier
                mediator_reward += success_bonus
                learning_bonus_applied += success_bonus
            elif steps_without_progress > 15:
                stagnation_penalty = 0.05
                mediator_reward -= stagnation_penalty

            # Learning exploration bonus (only in early phase)
            if learning_phase == "early_exploration":
                if was_interrupted and interaction_info.get('llm_plan_changed', False):
                    exploration_bonus = 0.05
                    mediator_reward += exploration_bonus
                    learning_bonus_applied += exploration_bonus

            tsc_agent.mediator.train_asking_policy(
                obs=llm_obs,
                action=rl_action,
                reward=mediator_reward,
                next_obs=llm_obs,
                asked_llm=was_interrupted,
                llm_plan_changed=interaction_info.get('llm_plan_changed', False)
            )

    # Update performance tracking
    tsc_agent.update_performance(episode_success)

    # Calculate learning-aware efficiency metrics
    interrupt_rate = interrupts / sim_step if sim_step > 0 else 0
    override_rate = overrides / max(interrupts, 1)
    agreement_rate = agreements / max(interrupts, 1)
    efficiency = overrides / max(interrupts, 1) if interrupts > 0 else 1.0

    # Learning phase aware success evaluation
    success_emoji = "✅" if total_reward > 0 else "❌"
    if learning_phase == "early_exploration":
        efficiency_emoji = "🎓" if efficiency > 0.3 else "📚" if efficiency > 0.1 else "❓"
    elif learning_phase == "guided_learning":
        efficiency_emoji = "🎯" if efficiency > 0.4 else "⚠️" if efficiency > 0.2 else "❌"
    else:  # autonomous
        efficiency_emoji = "🚀" if efficiency > 0.5 else "⚠️" if efficiency > 0.2 else "❌"

    logger.info(f"Episode {episode} [{learning_phase.upper()}] END: Reward={total_reward:.2f}, "
                f"Interrupts={interrupts}, Overrides={overrides}, Agreements={agreements}, "
                f"Efficiency={efficiency:.1%} {efficiency_emoji}, Success={success_emoji}, "
                f"Learning_Bonus={learning_bonus_applied:.3f}")

    return {
        'reward': total_reward,
        'steps': sim_step,
        'success': total_reward > 0,
        'interrupts': interrupts,
        'overrides': overrides,
        'agreements': agreements,
        'interrupt_rate': interrupt_rate,
        'override_rate': override_rate,
        'agreement_rate': agreement_rate,
        'efficiency': efficiency,
        'step_rewards': step_rewards,
        'avg_step_reward': np.mean(step_rewards) if step_rewards else 0,
        'episode_success': episode_success,
        'reward_per_step': total_reward / max(sim_step, 1),
        'learning_phase': learning_phase,
        'learning_bonus_applied': learning_bonus_applied
    }


def evaluate_baseline(rl_agent, rl_env, num_episodes: int = 10):
    """
    Evaluate baseline RL performance without LLM
    """
    logger.info("Evaluating baseline RL performance (no LLM)...")
    baseline_results = []

    for episode in range(num_episodes):
        obs, _ = rl_env.reset(seed=episode + 1000)
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 100:
            action, _ = rl_agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = rl_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        baseline_results.append({
            'reward': total_reward,
            'success': total_reward > 0,
            'steps': steps
        })

    baseline_success = np.mean([r['success'] for r in baseline_results])
    baseline_reward = np.mean([r['reward'] for r in baseline_results])
    logger.info(f"Baseline RL: Success={baseline_success:.1%}, Avg Reward={baseline_reward:.3f}")
    return baseline_success, baseline_reward


def main():
    """Enhanced experiment with learning-aware mediator training and adaptive thresholds."""

    # Enhanced WandB config
    wandb.init(
        project="TSC_Mediator_Learning_Aware",
        entity="BILGEM_DCS_RL",
        config={
            "env_name": "MiniGrid-DoorKey-6x6-v0",
            "mediator_episodes": 75,  # More episodes for learning phases
            "max_steps": 100,
            "algo": "TSC_Mediator_Learning_Aware",
            "llm_model": "llama3.1:8b",
            "mediator_lr": 1e-4,
            "baseline_episodes": 10,
            "mediator_hidden_dim": 64,
            "learning_phases": ["early_exploration", "guided_learning", "autonomous"],
            "early_exploration_episodes": 25,
            "guided_learning_episodes": 25,
            "autonomous_episodes": 25,
            "lambda_penalty_start": 0.01,  # More gentle start
            "lambda_penalty_end": 0.15,  # More gentle end
            "agreement_penalty_start": 0.05,  # More gentle
            "agreement_penalty_end": 0.15,  # More gentle
            "gradient_clip_early": 0.3,
            "gradient_clip_advanced": 0.2,
            "entropy_bonus_early": 0.03,
            "entropy_bonus_advanced": 0.02,
            "l2_regularization": 0.001,
            "learning_tolerance_multiplier": 2.0,
            "early_stopping_patience": 20,
        }
    )

    logger.info("Starting LEARNING-AWARE Mediator Experiment")
    logger.info("Focus: Gradual learning with phase-appropriate guidance and loop prevention")

    # Setup
    device = get_device()
    logger.info(f"Using device: {device}")

    # Initialize Llama
    try:
        chat = ChatOllama(
            model="llama3.1:8b",
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            num_predict=200,  # Slightly longer for educational responses
        )
        logger.info("🦙 Llama 3.1-8B model initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Llama model: {e}")
        logger.error("Make sure Ollama is running and llama3.1:8b model is downloaded:")
        logger.error("  1. Start Ollama: 'ollama serve'")
        logger.error("  2. Download model: 'ollama pull llama3.1:8b'")
        return

    llm = RunnableLambda(lambda x: chat.invoke(x))

    # Test Llama connection
    try:
        test_response = llm.invoke("Hello, respond with just 'OK'")
        logger.info(f"🦙 Llama test successful: {test_response}")
    except Exception as e:
        logger.error(f"❌ Llama connection failed: {e}")
        return

    # Initialize environments
    env_name = "MiniGrid-DoorKey-6x6-v0"
    rl_env, llm_env = make_env(env_name=env_name, max_steps=100)

    # Load RL Agent
    model_path = "models/ppo_minigrid_doorkey_6x6_250000_steps"

    try:
        rl_agent = PPO.load(model_path, device=device)
        logger.info("RL agent loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load RL agent: {e}")
        logger.info("Training environment will use random actions for demonstration")
        rl_agent = None

    # Initialize Learning-Aware TSC Agent
    obs_shape = llm_env.observation_space['image'].shape
    tsc_agent = TSCAgentWithMediator(
        llm=llm,
        obs_shape=obs_shape,
        device=device,
        verbose=True,
        train_mediator=True
    )

    logger.info("Setup complete - LEARNING-AWARE mediator with gradual phase transitions")

    # Baseline evaluation
    baseline_success = None
    baseline_reward = None
    if rl_agent is not None:
        baseline_success, baseline_reward = evaluate_baseline(rl_agent, rl_env, num_episodes=10)

        wandb.log({
            "baseline/success_rate": baseline_success,
            "baseline/avg_reward": baseline_reward,
        })

    # Learning-aware training
    num_episodes = 75  # Extended for learning phases
    logger.info(f"Training learning-aware mediator for {num_episodes} episodes...")

    results = []
    best_performance = -float('inf')
    patience = 20  # Increased patience for learning
    episodes_without_improvement = 0

    # Track learning phase transitions
    phase_transitions = []

    for episode in range(num_episodes):
        try:
            result = run_episode_flow(
                rl_agent=rl_agent,
                tsc_agent=tsc_agent,
                rl_env=rl_env,
                llm_env=llm_env,
                episode=episode
            )
            results.append(result)

            # Track phase transitions
            current_phase = result['learning_phase']
            if len(phase_transitions) == 0 or phase_transitions[-1] != current_phase:
                phase_transitions.append(current_phase)
                logger.info(f"🔄 Learning phase transition: {current_phase.upper()}")

            # Learning-aware performance tracking
            recent_performance = np.mean([r['reward'] for r in results[-10:]])
            recent_efficiency = np.mean([r['efficiency'] for r in results[-10:]])
            recent_learning_bonus = np.mean([r.get('learning_bonus_applied', 0) for r in results[-5:]])

            # Phase-appropriate performance metrics
            if current_phase == "early_exploration":
                # Focus on learning and exploration
                combined_metric = recent_performance * 0.6 + recent_efficiency * 0.2 + recent_learning_bonus * 0.2
            elif current_phase == "guided_learning":
                # Balance performance and efficiency
                combined_metric = recent_performance * 0.7 + recent_efficiency * 0.3
            else:  # autonomous
                # Focus on efficiency and performance
                combined_metric = recent_performance * 0.5 + recent_efficiency * 0.5

            # Save best model
            if combined_metric > best_performance:
                best_performance = combined_metric
                episodes_without_improvement = 0
                os.makedirs("models", exist_ok=True)
                save_name = f"models/learning_aware_mediator_{current_phase}_ep_{episode}.pt"
                tsc_agent.save_mediator(save_name)
                logger.info(
                    f"🎯 New best model saved! Phase={current_phase}, Performance={recent_performance:.3f}, "
                    f"Efficiency={recent_efficiency:.1%}, Combined={combined_metric:.3f}")
            else:
                episodes_without_improvement += 1

            # Learning-phase-aware early stopping
            if episodes_without_improvement > patience and episode > 30:
                if current_phase == "autonomous" and recent_efficiency < 0.2:
                    logger.warning(f"Early stopping in autonomous phase due to low efficiency: {recent_efficiency:.1%}")
                    break

            mediator_stats = tsc_agent.get_mediator_stats()

            # Enhanced logging with learning context
            log_data = {
                "episode": episode,
                "training/success": int(result['success']),
                "training/reward": result['reward'],
                "training/steps": result['steps'],
                "training/interrupts": result['interrupts'],
                "training/overrides": result['overrides'],
                "training/agreements": result['agreements'],
                "training/interrupt_rate": result['interrupt_rate'],
                "training/override_rate": result['override_rate'],
                "training/agreement_rate": result['agreement_rate'],
                "training/efficiency": result['efficiency'],
                "training/avg_step_reward": result['avg_step_reward'],
                "training/combined_metric": combined_metric,
                "training/learning_bonus": result.get('learning_bonus_applied', 0),

                # Learning phase metrics
                "learning/phase": {"early_exploration": 0, "guided_learning": 1, "autonomous": 2}[current_phase],
                "learning/recent_performance": recent_performance,
                "learning/recent_efficiency": recent_efficiency,
                "learning/recent_learning_bonus": recent_learning_bonus,
                "learning/tolerance_multiplier": tsc_agent.learning_tolerance_multiplier,
                "learning/episodes_without_improvement": episodes_without_improvement,

                # Enhanced mediator metrics
                "mediator/ask_rate": mediator_stats.get('recent_ask_rate', 0),
                "mediator/avg_reward": mediator_stats.get('recent_avg_reward', 0),
                "mediator/recent_loss": mediator_stats.get('recent_loss', 0),
                "mediator/lambda_penalty": mediator_stats.get('lambda_penalty', 0.01),
                "mediator/agreement_penalty": mediator_stats.get('agreement_penalty', 0.05),
                "mediator/interrupt_efficiency": mediator_stats.get('interrupt_efficiency', 0),
                "mediator/recent_interrupt_rate": mediator_stats.get('recent_interrupt_rate', 0),
                "mediator/recent_agreement_rate": mediator_stats.get('recent_agreement_rate', 0),
                "mediator/baseline_reward": mediator_stats.get('baseline_reward', 0),
                "mediator/forced_rl_mode_steps": mediator_stats.get('forced_rl_mode_steps', 0),
                "mediator/learning_progress": mediator_stats.get('learning_progress', 0),
            }

            wandb.log(log_data)

            # Enhanced progress logging every 5 episodes
            if episode % 5 == 0 and episode > 0:
                recent_results = results[-5:]
                avg_success = np.mean([r['success'] for r in recent_results])
                avg_efficiency = np.mean([r['efficiency'] for r in recent_results])
                avg_interrupt_rate = np.mean([r['interrupt_rate'] for r in recent_results])
                avg_agreement_rate = np.mean([r['agreement_rate'] for r in recent_results])
                avg_reward = np.mean([r['reward'] for r in recent_results])
                avg_learning_bonus = np.mean([r.get('learning_bonus_applied', 0) for r in recent_results])

                logger.info(f"Episode {episode} [{current_phase.upper()}]: Success={avg_success:.1%}, "
                            f"Task_Reward={avg_reward:.3f}, Efficiency={avg_efficiency:.1%}, "
                            f"Interrupt={avg_interrupt_rate:.1%}, Agreement={avg_agreement_rate:.1%}, "
                            f"Learning_Bonus={avg_learning_bonus:.3f}")

                # Phase-appropriate feedback
                if current_phase == "early_exploration":
                    if avg_learning_bonus > 0.05:
                        logger.info(f"🎓 GOOD LEARNING: High learning bonus indicates good exploration!")
                    elif avg_efficiency < 0.1:
                        logger.info(f"📚 LEARNING MODE: Low efficiency is expected during exploration")
                elif current_phase == "guided_learning":
                    if avg_efficiency > 0.3 and avg_success > 0.5:
                        logger.info(f"🎯 GOOD PROGRESS: Balancing efficiency and success well")
                    elif avg_agreement_rate > 0.7:
                        logger.info(f"⚖️ HIGH AGREEMENTS: Consider if mediator is being too conservative")
                else:  # autonomous
                    if avg_efficiency > 0.5:
                        logger.info(f"🚀 EXCELLENT AUTONOMY: High efficiency in autonomous mode")
                    elif avg_efficiency < 0.3:
                        logger.info(f"⚠️ EFFICIENCY WARNING: Need better decision making in autonomous mode")

                # Mediator internal metrics
                mediator_reward = mediator_stats.get('recent_avg_reward', 0)
                logger.info(f"Mediator: Ask_Rate={mediator_stats.get('recent_ask_rate', 0):.3f}, "
                            f"Mediator_Reward={mediator_reward:.3f}, "
                            f"λ={mediator_stats.get('lambda_penalty', 0.01):.3f}, "
                            f"Agreement_Penalty={mediator_stats.get('agreement_penalty', 0.05):.3f}, "
                            f"Forced_RL={mediator_stats.get('forced_rl_mode_steps', 0)}")

        except Exception as e:
            logger.error(f"Episode {episode} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comprehensive final results
    print("\n" + "=" * 80)
    print("LEARNING-AWARE MEDIATOR TRAINING RESULTS")
    print("=" * 80)

    if results:
        # Overall performance
        success_rate = np.mean([r['success'] for r in results])
        avg_reward = np.mean([r['reward'] for r in results])
        avg_efficiency = np.mean([r['efficiency'] for r in results])
        avg_interrupts = np.mean([r['interrupts'] for r in results])
        avg_interrupt_rate = np.mean([r['interrupt_rate'] for r in results])
        avg_override_rate = np.mean([r['override_rate'] for r in results])
        avg_agreement_rate = np.mean([r['agreement_rate'] for r in results])
        total_learning_bonus = sum([r.get('learning_bonus_applied', 0) for r in results])

        print(f"Enhanced TSC Agent Performance:")
        print(f"Success Rate:           {success_rate:.1%}")
        print(f"Average Task Reward:    {avg_reward:.3f}")
        print(f"Average Efficiency:     {avg_efficiency:.1%}")
        print(f"Average Interrupts:     {avg_interrupts:.1f} per episode")
        print(f"Average Interrupt Rate: {avg_interrupt_rate:.1%}")
        print(f"Average Override Rate:  {avg_override_rate:.1%}")
        print(f"Average Agreement Rate: {avg_agreement_rate:.1%}")
        print(f"Total Learning Bonus:   {total_learning_bonus:.3f}")

        # Phase-specific analysis
        print(f"\nLearning Phase Analysis:")
        for phase in ["early_exploration", "guided_learning", "autonomous"]:
            phase_results = [r for r in results if r.get('learning_phase') == phase]
            if phase_results:
                phase_success = np.mean([r['success'] for r in phase_results])
                phase_efficiency = np.mean([r['efficiency'] for r in phase_results])
                phase_learning_bonus = sum([r.get('learning_bonus_applied', 0) for r in phase_results])
                print(f"  {phase.upper()}: Success={phase_success:.1%}, "
                      f"Efficiency={phase_efficiency:.1%}, Learning_Bonus={phase_learning_bonus:.3f}")

        # Final efficiency analysis
        final_phase_results = results[-10:]  # Last 10 episodes
        final_efficiency = np.mean([r['efficiency'] for r in final_phase_results])
        final_success = np.mean([r['success'] for r in final_phase_results])

        if final_efficiency > 0.5 and final_success > 0.7:
            print(f"🚀 EXCELLENT FINAL PERFORMANCE: {final_efficiency:.1%} efficiency, {final_success:.1%} success")
        elif final_efficiency > 0.3 and final_success > 0.5:
            print(f"✅ GOOD FINAL PERFORMANCE: {final_efficiency:.1%} efficiency, {final_success:.1%} success")
        else:
            print(f"⚠️ NEEDS IMPROVEMENT: {final_efficiency:.1%} efficiency, {final_success:.1%} success")

        # Final metrics for wandB
        final_metrics = {
            "final/success_rate": success_rate,
            "final/avg_reward": avg_reward,
            "final/avg_efficiency": avg_efficiency,
            "final/avg_interrupts": avg_interrupts,
            "final/total_learning_bonus": total_learning_bonus,
            "final/phase_transitions": len(phase_transitions),
            "final/final_efficiency": final_efficiency,
            "final/final_success": final_success,
            "final/episodes_without_improvement": episodes_without_improvement,
            "final/best_combined_metric": best_performance,
        }

        if baseline_success is not None:
            improvement = success_rate - baseline_success
            reward_improvement = avg_reward - baseline_reward if baseline_reward else 0
            print(f"\nComparison to Baseline RL:")
            print(f"Baseline Success Rate:  {baseline_success:.1%}")
            print(f"Baseline Avg Reward:    {baseline_reward:.3f}")
            print(f"Success Improvement:    {improvement:+.1%}")
            print(f"Reward Improvement:     {reward_improvement:+.3f}")

            final_metrics.update({
                "final/baseline_success": baseline_success,
                "final/baseline_reward": baseline_reward,
                "final/success_improvement": improvement,
                "final/reward_improvement": reward_improvement
            })

        # Final mediator statistics
        mediator_stats = tsc_agent.get_mediator_stats()
        print(f"\nMediator Learning Progress:")
        print(f"Final Learning Phase:      {mediator_stats.get('learning_phase', 'unknown')}")
        print(f"Total Training Steps:      {mediator_stats.get('total_steps', 0)}")
        print(f"Current Ask Rate:          {mediator_stats.get('recent_ask_rate', 0):.3f}")
        print(f"Interrupt Efficiency:      {mediator_stats.get('interrupt_efficiency', 0):.1%}")
        print(f"Learning Progress:         {mediator_stats.get('learning_progress', 0):.1%}")
        print(f"Final λ Penalty:           {mediator_stats.get('lambda_penalty', 0.01):.3f}")
        print(f"Final Agreement Penalty:   {mediator_stats.get('agreement_penalty', 0.05):.3f}")
        print(f"Learning Tolerance Multi:  {tsc_agent.learning_tolerance_multiplier:.2f}")

        wandb.log(final_metrics)

        # Save final trained mediator
        save_path = "models/learning_aware_mediator_final.pt"
        os.makedirs("models", exist_ok=True)
        tsc_agent.save_mediator(save_path)
        logger.info(f"Saved final learning-aware mediator to {save_path}")

        # Save model as WandB artifact
        artifact = wandb.Artifact("learning_aware_mediator_model", type="model")
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)

        # Final assessment
        if final_success > 0.8 and final_efficiency > 0.5:
            logger.info("🎉 OUTSTANDING! High success with excellent efficiency!")
        elif final_success > 0.7 and final_efficiency > 0.3:
            logger.info("✅ EXCELLENT! Good performance with reasonable efficiency")
        elif success_rate > baseline_success if baseline_success else 0.5:
            logger.info("🔄 IMPROVING! Better than baseline, continue training recommended")
        else:
            logger.info("📚 LEARNING! Agent is still developing, consider extended training")

    else:
        logger.error("No successful episodes completed!")

    # Cleanup
    rl_env.close()
    llm_env.close()
    wandb.finish()


if __name__ == '__main__':
    main()