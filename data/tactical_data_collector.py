# Systematic Tactical Data Collection for Enhanced Training
import numpy as np
import json
import time
import os
import pickle
import random
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict


class ManeuverType(Enum):
    """Types of tactical maneuvers to collect data for"""
    STRAIGHT_LINE_ATTACK = "straight_line_attack"
    SERPENTINE_APPROACH = "serpentine_approach"
    CIRCULAR_ENGAGEMENT = "circular_engagement"
    DEFENSIVE_SPIRAL = "defensive_spiral"
    BVR_MISSILE_SHOT = "bvr_missile_shot"
    WVR_GUNS_ATTACK = "wvr_guns_attack"
    NOTCH_MANEUVER = "notch_maneuver"
    ENERGY_MANAGEMENT = "energy_management"
    MULTI_TARGET_ENGAGEMENT = "multi_target_engagement"


@dataclass
class TacticalTrajectoryPoint:
    """Single point in a tactical trajectory"""
    timestamp: float
    step: int
    
    # Aircraft state
    position: np.ndarray
    velocity: np.ndarray
    attitude: np.ndarray  # [pitch, roll, yaw]
    
    # Tactical state
    distance_to_target: float
    aspect_angle: float
    closure_rate: float
    energy_state: str
    threat_level: float
    
    # Action taken
    action: np.ndarray
    
    # Outcome
    reward: float
    success: bool
    
    # Context
    maneuver_type: ManeuverType
    enemy_behavior: str
    environmental_conditions: Dict[str, Any]


@dataclass
class TacticalTrajectory:
    """Complete tactical trajectory with metadata"""
    trajectory_id: str
    maneuver_type: ManeuverType
    start_time: float
    end_time: float
    total_steps: int
    
    # Trajectory data
    trajectory_points: List[TacticalTrajectoryPoint]
    
    # Performance metrics
    success: bool
    final_reward: float
    maneuver_effectiveness: float
    tactical_quality_score: float
    
    # Context
    initial_conditions: Dict[str, Any]
    enemy_configuration: Dict[str, Any]
    environmental_factors: Dict[str, Any]
    
    # Analysis
    key_decision_points: List[int]  # Step indices of critical decisions
    failure_analysis: Optional[Dict[str, Any]] = None


class TacticalDataCollector:
    """
    Systematic collector of tactical maneuver data for training enhancement
    and LLM fine-tuning. Collects expert demonstrations, successful maneuvers,
    and failure cases for comprehensive tactical knowledge.
    """
    
    def __init__(self, collection_dir: str = "data/tactical_collections"):
        """
        Initialize tactical data collector
        
        Args:
            collection_dir: Directory for storing collected data
        """
        self.collection_dir = collection_dir
        os.makedirs(collection_dir, exist_ok=True)
        
        # Collection state
        self.current_trajectory = None
        self.collected_trajectories = []
        self.collection_statistics = defaultdict(int)
        
        # Quality filters
        self.quality_thresholds = {
            'minimum_trajectory_length': 10,
            'minimum_success_rate': 0.1,  # Collect failures too
            'maximum_trajectory_length': 2000
        }
        
        # Maneuver-specific collection targets
        self.collection_targets = {
            ManeuverType.STRAIGHT_LINE_ATTACK: 200,
            ManeuverType.SERPENTINE_APPROACH: 150,
            ManeuverType.CIRCULAR_ENGAGEMENT: 150,
            ManeuverType.DEFENSIVE_SPIRAL: 100,
            ManeuverType.BVR_MISSILE_SHOT: 300,
            ManeuverType.WVR_GUNS_ATTACK: 200,
            ManeuverType.NOTCH_MANEUVER: 100,
            ManeuverType.ENERGY_MANAGEMENT: 250,
            ManeuverType.MULTI_TARGET_ENGAGEMENT: 100
        }
        
        print(f"[TACTICAL COLLECTOR] Initialized with {len(self.collection_targets)} maneuver types")
        print(f"[TACTICAL COLLECTOR] Target: {sum(self.collection_targets.values())} trajectories")
    
    def start_trajectory_collection(self, maneuver_type: ManeuverType,
                                  initial_conditions: Dict[str, Any],
                                  enemy_config: Dict[str, Any],
                                  environmental_factors: Dict[str, Any] = None) -> str:
        """
        Start collecting a new tactical trajectory
        
        Args:
            maneuver_type: Type of maneuver being performed
            initial_conditions: Initial tactical situation
            enemy_config: Enemy configuration
            environmental_factors: Environmental conditions
        
        Returns:
            Trajectory ID for reference
        """
        
        trajectory_id = f"{maneuver_type.value}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        self.current_trajectory = TacticalTrajectory(
            trajectory_id=trajectory_id,
            maneuver_type=maneuver_type,
            start_time=time.time(),
            end_time=0.0,
            total_steps=0,
            trajectory_points=[],
            success=False,
            final_reward=0.0,
            maneuver_effectiveness=0.0,
            tactical_quality_score=0.0,
            initial_conditions=initial_conditions.copy(),
            enemy_configuration=enemy_config.copy(),
            environmental_factors=environmental_factors.copy() if environmental_factors else {},
            key_decision_points=[]
        )
        
        print(f"[COLLECTION] Started trajectory: {trajectory_id} ({maneuver_type.value})")
        return trajectory_id
    
    def record_trajectory_point(self, step: int, aircraft_state: Dict[str, Any],
                              action: np.ndarray, reward: float, tactical_context: Dict[str, Any]):
        """
        Record a point in the current trajectory
        
        Args:
            step: Current step number
            aircraft_state: Complete aircraft state
            action: Action taken
            reward: Reward received
            tactical_context: Additional tactical context
        """
        
        if self.current_trajectory is None:
            print("[COLLECTION] No active trajectory - call start_trajectory_collection() first")
            return
        
        # Create trajectory point
        point = TacticalTrajectoryPoint(
            timestamp=time.time(),
            step=step,
            position=np.array(aircraft_state.get('position', [0, 0, 0])),
            velocity=np.array(aircraft_state.get('velocity', [0, 0, 0])),
            attitude=np.array(aircraft_state.get('attitude', [0, 0, 0])),
            distance_to_target=tactical_context.get('distance', 0),
            aspect_angle=tactical_context.get('aspect_angle', 0),
            closure_rate=tactical_context.get('closure_rate', 0),
            energy_state=tactical_context.get('energy_state', 'MEDIUM'),
            threat_level=tactical_context.get('threat_level', 0),
            action=action.copy(),
            reward=reward,
            success=tactical_context.get('success', False),
            maneuver_type=self.current_trajectory.maneuver_type,
            enemy_behavior=tactical_context.get('enemy_behavior', 'unknown'),
            environmental_conditions=tactical_context.get('environmental_conditions', {})
        )
        
        # Add to current trajectory
        self.current_trajectory.trajectory_points.append(point)
        self.current_trajectory.total_steps += 1
        
        # Detect key decision points
        if self._is_key_decision_point(point, tactical_context):
            self.current_trajectory.key_decision_points.append(step)
    
    def _is_key_decision_point(self, point: TacticalTrajectoryPoint, context: Dict[str, Any]) -> bool:
        """Identify key decision points in trajectory"""
        
        # High reward/penalty moments
        if abs(point.reward) > 5.0:
            return True
        
        # Lock acquisition/loss
        if context.get('lock_state_changed', False):
            return True
        
        # Missile firing
        if point.action[3] > 0.5:
            return True
        
        # High threat situations
        if point.threat_level > 0.7:
            return True
        
        # Phase transitions
        if context.get('phase_changed', False):
            return True
        
        return False
    
    def finish_trajectory_collection(self, success: bool, final_reward: float,
                                   failure_reason: str = None) -> bool:
        """
        Finish collecting current trajectory
        
        Args:
            success: Whether trajectory was successful
            final_reward: Final cumulative reward
            failure_reason: Reason for failure (if applicable)
        
        Returns:
            True if trajectory was saved, False if rejected
        """
        
        if self.current_trajectory is None:
            print("[COLLECTION] No active trajectory to finish")
            return False
        
        # Complete trajectory metadata
        self.current_trajectory.end_time = time.time()
        self.current_trajectory.success = success
        self.current_trajectory.final_reward = final_reward
        
        # Calculate trajectory quality metrics
        self._calculate_trajectory_quality()
        
        # Quality check
        if self._passes_quality_check():
            # Add failure analysis if unsuccessful
            if not success and failure_reason:
                self.current_trajectory.failure_analysis = {
                    'reason': failure_reason,
                    'failure_step': self.current_trajectory.total_steps,
                    'failure_conditions': self._analyze_failure_conditions()
                }
            
            # Save trajectory
            self._save_trajectory()
            
            # Update statistics
            self.collection_statistics[self.current_trajectory.maneuver_type] += 1
            self.collection_statistics['total_trajectories'] += 1
            
            print(f"[COLLECTION] Trajectory saved: {self.current_trajectory.trajectory_id} "
                  f"(success: {success}, quality: {self.current_trajectory.tactical_quality_score:.2f})")
            
            self.current_trajectory = None
            return True
        else:
            print(f"[COLLECTION] Trajectory rejected (quality check failed)")
            self.current_trajectory = None
            return False
    
    def _calculate_trajectory_quality(self):
        """Calculate overall quality of trajectory"""
        
        if not self.current_trajectory.trajectory_points:
            self.current_trajectory.tactical_quality_score = 0.0
            return
        
        points = self.current_trajectory.trajectory_points
        
        # Action smoothness
        action_changes = []
        for i in range(1, len(points)):
            change = np.linalg.norm(points[i].action[:3] - points[i-1].action[:3])
            action_changes.append(change)
        
        action_smoothness = 1.0 / (1.0 + np.mean(action_changes)) if action_changes else 1.0
        
        # Tactical consistency
        threat_responses = []
        for point in points:
            if point.threat_level > 0.6:
                # Check if action was appropriate for threat level
                action_magnitude = np.linalg.norm(point.action[:3])
                appropriate_response = action_magnitude > 0.3  # Should maneuver under threat
                threat_responses.append(appropriate_response)
        
        tactical_consistency = np.mean(threat_responses) if threat_responses else 0.5
        
        # Maneuver completion
        maneuver_completion = 1.0 if self.current_trajectory.success else 0.3
        
        # Overall quality score
        quality = (action_smoothness * 0.3 + 
                  tactical_consistency * 0.4 + 
                  maneuver_completion * 0.3)
        
        self.current_trajectory.tactical_quality_score = quality
        
        # Calculate maneuver effectiveness
        if self.current_trajectory.final_reward > 0:
            effectiveness = min(1.0, self.current_trajectory.final_reward / 100.0)
        else:
            effectiveness = max(0.0, 1.0 + self.current_trajectory.final_reward / 100.0)
        
        self.current_trajectory.maneuver_effectiveness = effectiveness
    
    def _passes_quality_check(self) -> bool:
        """Check if trajectory meets quality standards"""
        
        trajectory = self.current_trajectory
        
        # Length check
        if trajectory.total_steps < self.quality_thresholds['minimum_trajectory_length']:
            return False
        
        if trajectory.total_steps > self.quality_thresholds['maximum_trajectory_length']:
            return False
        
        # Quality score check
        if trajectory.tactical_quality_score < 0.2:  # Very low quality
            return False
        
        # Data completeness check
        if len(trajectory.trajectory_points) != trajectory.total_steps:
            return False
        
        return True
    
    def _analyze_failure_conditions(self) -> Dict[str, Any]:
        """Analyze conditions that led to trajectory failure"""
        
        if not self.current_trajectory.trajectory_points:
            return {'error': 'no_data'}
        
        points = self.current_trajectory.trajectory_points
        final_point = points[-1]
        
        # Analyze final conditions
        failure_analysis = {
            'final_distance': final_point.distance_to_target,
            'final_threat_level': final_point.threat_level,
            'final_energy_state': final_point.energy_state,
            'final_action': final_point.action.tolist(),
            'trajectory_length': len(points),
            'average_reward': np.mean([p.reward for p in points]),
            'threat_exposure': np.mean([p.threat_level for p in points])
        }
        
        # Identify likely failure cause
        if final_point.distance_to_target < 1000:
            failure_analysis['likely_cause'] = 'collision_or_overshoot'
        elif final_point.threat_level > 0.8:
            failure_analysis['likely_cause'] = 'destroyed_by_enemy'
        elif np.mean([p.reward for p in points]) < -1.0:
            failure_analysis['likely_cause'] = 'poor_tactical_decisions'
        else:
            failure_analysis['likely_cause'] = 'unknown'
        
        return failure_analysis
    
    def _save_trajectory(self):
        """Save trajectory to disk"""
        
        trajectory = self.current_trajectory
        
        # Create maneuver-specific directory
        maneuver_dir = os.path.join(self.collection_dir, trajectory.maneuver_type.value)
        os.makedirs(maneuver_dir, exist_ok=True)
        
        # Save as pickle for full data
        pickle_path = os.path.join(maneuver_dir, f"{trajectory.trajectory_id}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(trajectory, f)
        
        # Save as JSON for analysis (simplified)
        json_data = {
            'trajectory_id': trajectory.trajectory_id,
            'maneuver_type': trajectory.maneuver_type.value,
            'success': trajectory.success,
            'final_reward': trajectory.final_reward,
            'tactical_quality_score': trajectory.tactical_quality_score,
            'total_steps': trajectory.total_steps,
            'duration': trajectory.end_time - trajectory.start_time,
            'initial_conditions': trajectory.initial_conditions,
            'key_decision_points': trajectory.key_decision_points,
            'failure_analysis': trajectory.failure_analysis
        }
        
        json_path = os.path.join(maneuver_dir, f"{trajectory.trajectory_id}.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def collect_expert_demonstrations(self, expert_agent, env, maneuver_type: ManeuverType,
                                    num_demonstrations: int = 50) -> List[str]:
        """
        Collect expert demonstrations for specific maneuver type
        
        Args:
            expert_agent: Expert agent (rule-based or trained)
            env: Environment for demonstration
            maneuver_type: Type of maneuver to demonstrate
            num_demonstrations: Number of demonstrations to collect
        
        Returns:
            List of collected trajectory IDs
        """
        
        print(f"[EXPERT COLLECTION] Collecting {num_demonstrations} {maneuver_type.value} demonstrations")
        
        collected_ids = []
        
        for demo in range(num_demonstrations):
            # Setup environment for specific maneuver
            initial_conditions = self._generate_maneuver_initial_conditions(maneuver_type)
            
            # Configure environment
            if hasattr(env, 'set_scenario_config'):
                env.set_scenario_config(initial_conditions)
            
            # Reset environment
            obs, info = env.reset(seed=demo)
            
            # Start trajectory collection
            trajectory_id = self.start_trajectory_collection(
                maneuver_type=maneuver_type,
                initial_conditions=initial_conditions,
                enemy_config={'skill': 'expert', 'behavior': 'tactical'},
                environmental_factors={'weather': 'clear', 'time': 'day'}
            )
            
            # Run demonstration
            done = False
            step = 0
            total_reward = 0
            
            while not done and step < 500:  # Max 500 steps per demo
                # Expert action
                if hasattr(expert_agent, 'predict'):
                    action, _ = expert_agent.predict(obs, deterministic=True)
                elif hasattr(expert_agent, 'get_action'):
                    action = expert_agent.get_action(obs)
                else:
                    action = env.action_space.sample()  # Fallback
                
                # Environment step
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                # Record trajectory point
                self.record_trajectory_point(
                    step=step,
                    aircraft_state={
                        'position': info.get('position', [0, 0, 5000]),
                        'velocity': info.get('velocity', [200, 0, 0]),
                        'attitude': info.get('attitude', [0, 0, 0])
                    },
                    action=action,
                    reward=reward,
                    tactical_context={
                        'distance': info.get('distance', 8000),
                        'aspect_angle': info.get('aspect_angle', 0),
                        'closure_rate': info.get('closure_rate', 0),
                        'energy_state': info.get('energy_state', 'MEDIUM'),
                        'threat_level': info.get('threat_level', 0.3),
                        'success': info.get('success', False),
                        'enemy_behavior': 'expert',
                        'environmental_conditions': {'weather': 'clear'}
                    }
                )
                
                obs = next_obs
                step += 1
            
            # Finish trajectory
            success = info.get('success', total_reward > 50)
            saved = self.finish_trajectory_collection(success, total_reward)
            
            if saved:
                collected_ids.append(trajectory_id)
            
            if demo % 10 == 0:
                print(f"   Progress: {demo+1}/{num_demonstrations} demonstrations")
        
        print(f"[EXPERT COLLECTION] Collected {len(collected_ids)} high-quality demonstrations")
        return collected_ids
    
    def _generate_maneuver_initial_conditions(self, maneuver_type: ManeuverType) -> Dict[str, Any]:
        """Generate appropriate initial conditions for maneuver type"""
        
        base_conditions = {
            'ego_position': [0, 0, 5000 + random.uniform(-1000, 1000)],
            'ego_velocity': [250 + random.uniform(-50, 50), 0, 0],
            'ego_heading': random.uniform(0, 360)
        }
        
        if maneuver_type == ManeuverType.BVR_MISSILE_SHOT:
            base_conditions.update({
                'enemy_position': [15000 + random.uniform(-3000, 3000), 
                                 random.uniform(-5000, 5000), 
                                 5000 + random.uniform(-1000, 1000)],
                'initial_lock': True,
                'missile_available': True
            })
        
        elif maneuver_type == ManeuverType.WVR_GUNS_ATTACK:
            base_conditions.update({
                'enemy_position': [3000 + random.uniform(-1000, 1000),
                                 random.uniform(-1000, 1000),
                                 5000 + random.uniform(-500, 500)],
                'initial_lock': False,
                'close_range': True
            })
        
        elif maneuver_type == ManeuverType.DEFENSIVE_SPIRAL:
            base_conditions.update({
                'enemy_position': [2000 + random.uniform(-500, 500),
                                 random.uniform(-1000, 1000),
                                 6000 + random.uniform(-500, 500)],
                'under_attack': True,
                'threat_level': 0.8
            })
        
        elif maneuver_type == ManeuverType.SERPENTINE_APPROACH:
            base_conditions.update({
                'enemy_position': [8000 + random.uniform(-2000, 2000),
                                 random.uniform(-3000, 3000),
                                 5000 + random.uniform(-1000, 1000)],
                'approach_required': True,
                'evasive_enemy': True
            })
        
        return base_conditions
    
    def analyze_collected_data(self) -> Dict[str, Any]:
        """Analyze all collected tactical data"""
        
        print(f"[ANALYSIS] Analyzing {len(self.collected_trajectories)} collected trajectories")
        
        analysis = {
            'collection_summary': dict(self.collection_statistics),
            'maneuver_analysis': {},
            'quality_metrics': {},
            'tactical_insights': {},
            'training_recommendations': []
        }
        
        # Group trajectories by maneuver type
        maneuver_groups = defaultdict(list)
        for trajectory in self.collected_trajectories:
            maneuver_groups[trajectory.maneuver_type].append(trajectory)
        
        # Analyze each maneuver type
        for maneuver_type, trajectories in maneuver_groups.items():
            maneuver_analysis = self._analyze_maneuver_type(trajectories)
            analysis['maneuver_analysis'][maneuver_type.value] = maneuver_analysis
        
        # Overall quality metrics
        if self.collected_trajectories:
            all_scores = [t.tactical_quality_score for t in self.collected_trajectories]
            all_rewards = [t.final_reward for t in self.collected_trajectories]
            
            analysis['quality_metrics'] = {
                'average_quality_score': np.mean(all_scores),
                'quality_std': np.std(all_scores),
                'average_reward': np.mean(all_rewards),
                'success_rate': np.mean([t.success for t in self.collected_trajectories]),
                'total_trajectories': len(self.collected_trajectories)
            }
        
        # Generate training recommendations
        analysis['training_recommendations'] = self._generate_training_recommendations(analysis)
        
        return analysis
    
    def _analyze_maneuver_type(self, trajectories: List[TacticalTrajectory]) -> Dict[str, Any]:
        """Analyze trajectories for specific maneuver type"""
        
        if not trajectories:
            return {'error': 'no_trajectories'}
        
        # Success analysis
        successful = [t for t in trajectories if t.success]
        failed = [t for t in trajectories if not t.success]
        
        analysis = {
            'total_trajectories': len(trajectories),
            'successful_trajectories': len(successful),
            'failed_trajectories': len(failed),
            'success_rate': len(successful) / len(trajectories),
            'average_quality': np.mean([t.tactical_quality_score for t in trajectories]),
            'average_effectiveness': np.mean([t.maneuver_effectiveness for t in trajectories])
        }
        
        # Compare successful vs failed trajectories
        if successful and failed:
            analysis['success_vs_failure'] = {
                'successful_avg_quality': np.mean([t.tactical_quality_score for t in successful]),
                'failed_avg_quality': np.mean([t.tactical_quality_score for t in failed]),
                'successful_avg_steps': np.mean([t.total_steps for t in successful]),
                'failed_avg_steps': np.mean([t.total_steps for t in failed])
            }
        
        # Identify common failure patterns
        if failed:
            failure_reasons = [t.failure_analysis.get('reason', 'unknown') for t in failed if t.failure_analysis]
            failure_counts = defaultdict(int)
            for reason in failure_reasons:
                failure_counts[reason] += 1
            analysis['common_failure_patterns'] = dict(failure_counts)
        
        return analysis
    
    def _generate_training_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on collected data analysis"""
        
        recommendations = []
        
        quality_metrics = analysis.get('quality_metrics', {})
        avg_quality = quality_metrics.get('average_quality_score', 0)
        success_rate = quality_metrics.get('success_rate', 0)
        
        # Quality-based recommendations
        if avg_quality < 0.5:
            recommendations.append("Low trajectory quality - focus on action smoothness training")
        
        if success_rate < 0.3:
            recommendations.append("Low success rate - implement curriculum learning")
        
        # Maneuver-specific recommendations
        maneuver_analysis = analysis.get('maneuver_analysis', {})
        
        for maneuver, data in maneuver_analysis.items():
            if data.get('success_rate', 1.0) < 0.4:
                recommendations.append(f"Poor {maneuver} performance - increase training focus")
        
        # Data collection recommendations
        total_trajectories = quality_metrics.get('total_trajectories', 0)
        if total_trajectories < 1000:
            recommendations.append("Collect more tactical data for robust training")
        
        return recommendations
    
    def export_training_dataset(self, output_path: str = "data/tactical_training_dataset.jsonl") -> str:
        """Export collected data as training dataset"""
        
        if not self.collected_trajectories:
            print("[EXPORT] No trajectories to export")
            return ""
        
        print(f"[EXPORT] Exporting {len(self.collected_trajectories)} trajectories to training dataset")
        
        # Convert trajectories to training examples
        training_examples = []
        
        for trajectory in self.collected_trajectories:
            # Extract key decision points for training
            for decision_step in trajectory.key_decision_points:
                if decision_step < len(trajectory.trajectory_points):
                    point = trajectory.trajectory_points[decision_step]
                    
                    # Create training example
                    example = {
                        'maneuver_type': trajectory.maneuver_type.value,
                        'tactical_situation': {
                            'distance': point.distance_to_target,
                            'aspect_angle': point.aspect_angle,
                            'closure_rate': point.closure_rate,
                            'energy_state': point.energy_state,
                            'threat_level': point.threat_level
                        },
                        'expert_action': point.action.tolist(),
                        'reward': point.reward,
                        'success': point.success,
                        'quality_score': trajectory.tactical_quality_score
                    }
                    
                    training_examples.append(example)
        
        # Save training dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"[EXPORT] Training dataset saved: {output_path} ({len(training_examples)} examples)")
        return output_path
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get current collection status"""
        
        status = {
            'total_collected': self.collection_statistics['total_trajectories'],
            'collection_targets': self.collection_targets,
            'progress_by_maneuver': {},
            'completion_percentage': 0.0
        }
        
        # Calculate progress for each maneuver type
        total_target = sum(self.collection_targets.values())
        total_collected = 0
        
        for maneuver_type, target in self.collection_targets.items():
            collected = self.collection_statistics[maneuver_type]
            total_collected += collected
            
            status['progress_by_maneuver'][maneuver_type.value] = {
                'collected': collected,
                'target': target,
                'progress': collected / target if target > 0 else 0
            }
        
        status['completion_percentage'] = total_collected / total_target if total_target > 0 else 0
        
        return status


class AutomaticDataCollectionSystem:
    """
    Automatic system for collecting tactical data during normal training
    """
    
    def __init__(self, collector: TacticalDataCollector, collection_probability: float = 0.1):
        """
        Initialize automatic collection system
        
        Args:
            collector: Tactical data collector
            collection_probability: Probability of collecting each episode
        """
        self.collector = collector
        self.collection_probability = collection_probability
        self.episodes_processed = 0
        self.episodes_collected = 0
        
        print(f"[AUTO COLLECTOR] Initialized with {collection_probability:.1%} collection rate")
    
    def maybe_start_collection(self, episode_info: Dict[str, Any]) -> Optional[str]:
        """Maybe start collecting trajectory based on probability and conditions"""
        
        self.episodes_processed += 1
        
        # Random collection decision
        if random.random() > self.collection_probability:
            return None
        
        # Determine maneuver type from episode conditions
        maneuver_type = self._infer_maneuver_type(episode_info)
        
        # Check if we need more data for this maneuver type
        if self._needs_more_data(maneuver_type):
            trajectory_id = self.collector.start_trajectory_collection(
                maneuver_type=maneuver_type,
                initial_conditions=episode_info.get('initial_conditions', {}),
                enemy_config=episode_info.get('enemy_config', {}),
                environmental_factors=episode_info.get('environmental_factors', {})
            )
            
            self.episodes_collected += 1
            return trajectory_id
        
        return None
    
    def _infer_maneuver_type(self, episode_info: Dict[str, Any]) -> ManeuverType:
        """Infer maneuver type from episode information"""
        
        # Simple inference based on initial conditions
        initial_distance = episode_info.get('initial_conditions', {}).get('distance', 8000)
        enemy_behavior = episode_info.get('enemy_config', {}).get('behavior', 'standard')
        
        if initial_distance > 15000:
            return ManeuverType.BVR_MISSILE_SHOT
        elif initial_distance < 3000:
            return ManeuverType.WVR_GUNS_ATTACK
        elif 'serpentine' in enemy_behavior:
            return ManeuverType.SERPENTINE_APPROACH
        elif 'circular' in enemy_behavior:
            return ManeuverType.CIRCULAR_ENGAGEMENT
        elif episode_info.get('under_attack', False):
            return ManeuverType.DEFENSIVE_SPIRAL
        else:
            return ManeuverType.STRAIGHT_LINE_ATTACK
    
    def _needs_more_data(self, maneuver_type: ManeuverType) -> bool:
        """Check if more data is needed for maneuver type"""
        
        collected = self.collector.collection_statistics[maneuver_type]
        target = self.collector.collection_targets[maneuver_type]
        
        return collected < target
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get automatic collection summary"""
        
        return {
            'episodes_processed': self.episodes_processed,
            'episodes_collected': self.episodes_collected,
            'collection_rate': self.episodes_collected / max(self.episodes_processed, 1),
            'target_collection_rate': self.collection_probability,
            'collector_status': self.collector.get_collection_status()
        }


if __name__ == "__main__":
    print("Tactical Data Collection System")
    
    # Create collector
    collector = TacticalDataCollector()
    
    # Test trajectory collection
    trajectory_id = collector.start_trajectory_collection(
        ManeuverType.BVR_MISSILE_SHOT,
        {'distance': 12000, 'locked': True},
        {'skill': 'expert'},
        {'weather': 'clear'}
    )
    
    # Simulate some trajectory points
    for step in range(20):
        collector.record_trajectory_point(
            step=step,
            aircraft_state={'position': [0, 0, 5000], 'velocity': [250, 0, 0], 'attitude': [0, 0, 0]},
            action=np.random.uniform(-1, 1, 4),
            reward=random.uniform(-1, 5),
            tactical_context={
                'distance': 12000 - step * 200,
                'threat_level': 0.3,
                'energy_state': 'HIGH'
            }
        )
    
    # Finish trajectory
    collector.finish_trajectory_collection(success=True, final_reward=50.0)
    
    print("Tactical data collection system ready!")
