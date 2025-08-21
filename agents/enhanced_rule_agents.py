# Enhanced Rule-Based Agents for Baseline Comparison and Expert Demonstrations
import numpy as np
import math
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque


class TacticalState(Enum):
    """Tactical states for rule-based decision making"""
    BVR_SEARCH = "bvr_search"
    BVR_LOCK = "bvr_lock"
    BVR_SHOT = "bvr_shot"
    MERGE_APPROACH = "merge_approach"
    WVR_MANEUVER = "wvr_maneuver"
    WVR_GUNS = "wvr_guns"
    DEFENSIVE = "defensive"
    EVASIVE = "evasive"


@dataclass
class PIDController:
    """PID controller for smooth aircraft control"""
    kp: float
    ki: float
    kd: float
    integral: float = 0.0
    prev_error: float = 0.0
    min_output: float = -1.0
    max_output: float = 1.0
    
    def update(self, error: float, dt: float = 1.0) -> float:
        """Update PID controller"""
        self.integral += error * dt
        self.integral = np.clip(self.integral, -0.5, 0.5)  # Prevent windup
        
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return np.clip(output, self.min_output, self.max_output)
    
    def reset(self):
        """Reset PID controller state"""
        self.integral = 0.0
        self.prev_error = 0.0


class EnhancedRuleAgent:
    """
    Enhanced rule-based agent with sophisticated tactical decision making,
    PID control, and real fighter pilot doctrine implementation
    """
    
    def __init__(self, agent_type: str = "expert", skill_level: float = 0.8):
        """
        Initialize enhanced rule agent
        
        Args:
            agent_type: Type of rule agent (novice, competent, expert, ace)
            skill_level: Skill level 0.0-1.0
        """
        self.agent_type = agent_type
        self.skill_level = skill_level
        
        # PID controllers for smooth flight
        self.pitch_controller = PIDController(kp=0.6, ki=0.05, kd=0.1, min_output=-0.8, max_output=0.8)
        self.roll_controller = PIDController(kp=0.8, ki=0.02, kd=0.15, min_output=-1.0, max_output=1.0)
        self.yaw_controller = PIDController(kp=0.4, ki=0.01, kd=0.05, min_output=-0.6, max_output=0.6)
        
        # Tactical state
        self.current_tactical_state = TacticalState.BVR_SEARCH
        self.state_timer = 0
        self.last_action = np.array([0.0, 0.0, 0.0, 0.0])
        self.decision_history = []
        
        # Tactical parameters based on skill level
        self.reaction_time = max(1, int(5 - skill_level * 4))  # 1-5 steps reaction time
        self.accuracy_factor = 0.5 + skill_level * 0.5  # 0.5-1.0 accuracy
        self.tactical_awareness = skill_level
        
        # Combat doctrine parameters
        self.doctrine = self._load_combat_doctrine(agent_type)
        
        print(f"[ENHANCED RULE] {agent_type} agent initialized (skill: {skill_level:.1f})")
        print(f"[ENHANCED RULE] Reaction time: {self.reaction_time} steps")
    
    def _load_combat_doctrine(self, agent_type: str) -> Dict[str, Any]:
        """Load combat doctrine parameters"""
        
        doctrines = {
            'novice': {
                'bvr_engagement_range': (8000, 20000),
                'wvr_engagement_range': (1000, 4000),
                'preferred_altitude': 5000,
                'energy_management_priority': 0.3,
                'threat_reaction_threshold': 0.8,
                'firing_discipline': 0.4
            },
            'competent': {
                'bvr_engagement_range': (10000, 18000),
                'wvr_engagement_range': (1500, 5000),
                'preferred_altitude': 7000,
                'energy_management_priority': 0.6,
                'threat_reaction_threshold': 0.6,
                'firing_discipline': 0.7
            },
            'expert': {
                'bvr_engagement_range': (12000, 16000),
                'wvr_engagement_range': (2000, 4000),
                'preferred_altitude': 8000,
                'energy_management_priority': 0.8,
                'threat_reaction_threshold': 0.4,
                'firing_discipline': 0.9
            },
            'ace': {
                'bvr_engagement_range': (13000, 15000),
                'wvr_engagement_range': (2500, 3500),
                'preferred_altitude': 9000,
                'energy_management_priority': 0.9,
                'threat_reaction_threshold': 0.3,
                'firing_discipline': 0.95
            }
        }
        
        return doctrines.get(agent_type, doctrines['competent'])
    
    def get_action(self, observation: np.ndarray, info: Dict[str, Any] = None) -> np.ndarray:
        """
        Get action based on enhanced rule-based decision making
        
        Args:
            observation: 25-dimensional tactical observation
            info: Additional environment information
        
        Returns:
            Action [pitch, roll, yaw, fire]
        """
        
        # Extract tactical features from observation
        features = self._extract_features(observation, info)
        
        # Update tactical state
        self._update_tactical_state(features)
        
        # Make tactical decision
        tactical_decision = self._make_tactical_decision(features)
        
        # Convert to control inputs using PID controllers
        control_action = self._tactical_to_control(tactical_decision, features)
        
        # Apply skill-based noise and limitations
        final_action = self._apply_skill_effects(control_action, features)
        
        # Record decision for analysis
        self._record_decision(features, tactical_decision, final_action)
        
        self.last_action = final_action
        return final_action
    
    def _extract_features(self, observation: np.ndarray, info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract tactical features from observation"""
        
        # Basic geometric features [0-12]
        dx, dy, dz = observation[0], observation[1], observation[2]
        distance = math.sqrt(dx**2 + dy**2 + dz**2) * 10000  # Denormalize
        
        ego_euler = observation[3:6]
        target_angle = observation[6]
        locked = observation[7] > 0
        missile_available = observation[8] > 0
        enemy_euler = observation[9:12]
        enemy_health = observation[12]
        
        # Enhanced tactical features [13-24] if available
        if len(observation) >= 25:
            closure_rate = observation[13] * 1000.0
            aspect_angle = observation[14] * 180.0
            g_force = observation[15] * 9.0
            turn_rate = observation[16] * 180.0
            climb_rate = observation[17] * 100.0
            threat_level = observation[18]
            lock_duration = observation[19] * 30.0
            time_since_lock = observation[20] * 100.0
            high_energy = bool(observation[21])
            low_energy = bool(observation[22])
            wvr_engagement = bool(observation[23])
            bvr_engagement = bool(observation[24])
        else:
            # Fallback for basic observation
            closure_rate = 0.0
            aspect_angle = abs(target_angle) * 180.0
            g_force = 1.0
            turn_rate = 0.0
            climb_rate = 0.0
            threat_level = 0.3
            lock_duration = 0.0
            time_since_lock = 0.0
            high_energy = distance > 8000
            low_energy = distance < 3000
            wvr_engagement = distance < 5000
            bvr_engagement = distance > 15000
        
        return {
            'distance': distance,
            'target_angle': target_angle,
            'locked': locked,
            'missile_available': missile_available,
            'enemy_health': enemy_health,
            'closure_rate': closure_rate,
            'aspect_angle': aspect_angle,
            'g_force': g_force,
            'turn_rate': turn_rate,
            'climb_rate': climb_rate,
            'threat_level': threat_level,
            'lock_duration': lock_duration,
            'high_energy': high_energy,
            'low_energy': low_energy,
            'wvr_engagement': wvr_engagement,
            'bvr_engagement': bvr_engagement,
            'ego_euler': ego_euler,
            'enemy_euler': enemy_euler
        }
    
    def _update_tactical_state(self, features: Dict[str, Any]):
        """Update current tactical state based on situation"""
        
        distance = features['distance']
        locked = features['locked']
        threat_level = features['threat_level']
        missile_available = features['missile_available']
        
        self.state_timer += 1
        
        # State transition logic
        if threat_level > 0.7:
            self.current_tactical_state = TacticalState.DEFENSIVE
        elif threat_level > 0.5:
            self.current_tactical_state = TacticalState.EVASIVE
        elif distance > 15000:
            if locked:
                self.current_tactical_state = TacticalState.BVR_SHOT
            else:
                self.current_tactical_state = TacticalState.BVR_SEARCH
        elif distance > 8000:
            if locked:
                self.current_tactical_state = TacticalState.BVR_LOCK
            else:
                self.current_tactical_state = TacticalState.MERGE_APPROACH
        elif distance > 3000:
            self.current_tactical_state = TacticalState.WVR_MANEUVER
        else:
            self.current_tactical_state = TacticalState.WVR_GUNS
        
        # Add state transition delays based on skill
        if self.state_timer < self.reaction_time:
            # Maintain previous state during reaction time
            pass
        else:
            self.state_timer = 0
    
    def _make_tactical_decision(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make high-level tactical decision"""
        
        state = self.current_tactical_state
        doctrine = self.doctrine
        
        decision = {
            'maneuver_type': 'maintain_course',
            'target_pitch': 0.0,
            'target_roll': 0.0,
            'target_yaw': 0.0,
            'fire_decision': False,
            'priority': 'low'
        }
        
        if state == TacticalState.BVR_SEARCH:
            # BVR search pattern
            decision.update({
                'maneuver_type': 'search_pattern',
                'target_pitch': 0.1,  # Slight climb for energy
                'target_roll': 0.0,
                'target_yaw': -features['target_angle'] * 0.5,  # Turn toward target
                'priority': 'medium'
            })
        
        elif state == TacticalState.BVR_LOCK:
            # Maintain lock and prepare for shot
            decision.update({
                'maneuver_type': 'lock_maintenance',
                'target_pitch': -features['target_angle'] * 0.8,  # Pitch to target
                'target_roll': 0.0,
                'target_yaw': -features['target_angle'] * 0.6,   # Yaw to target
                'priority': 'high'
            })
        
        elif state == TacticalState.BVR_SHOT:
            # Fire BVR missile
            optimal_range = doctrine['bvr_engagement_range']
            in_range = optimal_range[0] <= features['distance'] <= optimal_range[1]
            
            decision.update({
                'maneuver_type': 'bvr_engagement',
                'target_pitch': -features['target_angle'] * 0.5,
                'target_roll': 0.0,
                'target_yaw': -features['target_angle'] * 0.4,
                'fire_decision': (in_range and features['locked'] and 
                                features['missile_available'] and
                                random.random() < doctrine['firing_discipline']),
                'priority': 'critical'
            })
        
        elif state == TacticalState.WVR_MANEUVER:
            # WVR maneuvering for position
            decision.update({
                'maneuver_type': 'wvr_maneuvering',
                'target_pitch': self._calculate_wvr_pitch(features),
                'target_roll': self._calculate_wvr_roll(features),
                'target_yaw': self._calculate_wvr_yaw(features),
                'priority': 'high'
            })
        
        elif state == TacticalState.WVR_GUNS:
            # Guns tracking
            decision.update({
                'maneuver_type': 'guns_tracking',
                'target_pitch': -features['target_angle'] * 1.2,  # Aggressive tracking
                'target_roll': 0.0,
                'target_yaw': -features['target_angle'] * 0.8,
                'fire_decision': (abs(features['target_angle']) < 0.1 and 
                                features['distance'] < 2000),
                'priority': 'critical'
            })
        
        elif state == TacticalState.DEFENSIVE:
            # Defensive maneuvering
            decision.update({
                'maneuver_type': 'defensive_spiral',
                'target_pitch': -0.4,  # Dive
                'target_roll': 0.8 * math.sin(self.state_timer * 0.3),  # Rolling
                'target_yaw': 0.6 * math.cos(self.state_timer * 0.2),   # Turning
                'priority': 'critical'
            })
        
        elif state == TacticalState.EVASIVE:
            # Evasive maneuvering
            decision.update({
                'maneuver_type': 'evasive_maneuvers',
                'target_pitch': 0.3 * math.sin(self.state_timer * 0.4),
                'target_roll': 0.7 * math.cos(self.state_timer * 0.5),
                'target_yaw': 0.5 * math.sin(self.state_timer * 0.3),
                'priority': 'high'
            })
        
        return decision
    
    def _calculate_wvr_pitch(self, features: Dict[str, Any]) -> float:
        """Calculate optimal pitch for WVR combat"""
        
        # Energy management in WVR
        if features['high_energy']:
            # Use energy for aggressive maneuvering
            return -features['target_angle'] * 0.8
        elif features['low_energy']:
            # Conserve energy, climb if possible
            return 0.3
        else:
            # Balanced approach
            return -features['target_angle'] * 0.6
    
    def _calculate_wvr_roll(self, features: Dict[str, Any]) -> float:
        """Calculate optimal roll for WVR combat"""
        
        # Use roll for rapid direction changes
        target_angle = features['target_angle']
        
        if abs(target_angle) > 0.3:  # Target not in front
            # Roll toward target
            return np.sign(target_angle) * 0.7
        else:
            # Small roll for tracking
            return target_angle * 2.0
    
    def _calculate_wvr_yaw(self, features: Dict[str, Any]) -> float:
        """Calculate optimal yaw for WVR combat"""
        
        # Coordinated turn
        return -features['target_angle'] * 0.5
    
    def _tactical_to_control(self, tactical_decision: Dict[str, Any], 
                           features: Dict[str, Any]) -> np.ndarray:
        """Convert tactical decision to control inputs using PID"""
        
        # Target control positions
        target_pitch = tactical_decision['target_pitch']
        target_roll = tactical_decision['target_roll'] 
        target_yaw = tactical_decision['target_yaw']
        
        # Current control positions (from last action)
        current_pitch = self.last_action[0]
        current_roll = self.last_action[1]
        current_yaw = self.last_action[2]
        
        # Calculate errors
        pitch_error = target_pitch - current_pitch
        roll_error = target_roll - current_roll
        yaw_error = target_yaw - current_yaw
        
        # PID control
        pitch_cmd = self.pitch_controller.update(pitch_error)
        roll_cmd = self.roll_controller.update(roll_error)
        yaw_cmd = self.yaw_controller.update(yaw_error)
        
        # Fire decision
        fire_cmd = 1.0 if tactical_decision['fire_decision'] else 0.0
        
        return np.array([pitch_cmd, roll_cmd, yaw_cmd, fire_cmd])
    
    def _apply_skill_effects(self, action: np.ndarray, features: Dict[str, Any]) -> np.ndarray:
        """Apply skill-based effects to action"""
        
        skilled_action = action.copy()
        
        # Add skill-based noise (higher skill = less noise)
        noise_level = 0.1 * (1.0 - self.skill_level)
        noise = np.random.normal(0, noise_level, 3)
        skilled_action[:3] += noise
        
        # Apply accuracy factor to targeting
        skilled_action[0] *= self.accuracy_factor  # Pitch accuracy
        skilled_action[2] *= self.accuracy_factor  # Yaw accuracy
        
        # Skill-based firing decision modification
        if skilled_action[3] > 0.5:
            # Check firing conditions with skill-based probability
            fire_probability = self.skill_level * self.doctrine['firing_discipline']
            
            # Additional checks for good shots
            if features['locked'] and features['distance'] < 12000:
                fire_probability += 0.2
            
            if random.random() > fire_probability:
                skilled_action[3] = 0.0  # Don't fire
        
        # Clamp to action space
        skilled_action[:3] = np.clip(skilled_action[:3], -1.0, 1.0)
        skilled_action[3] = np.clip(skilled_action[3], 0.0, 1.0)
        
        return skilled_action
    
    def _record_decision(self, features: Dict[str, Any], tactical_decision: Dict[str, Any], 
                        final_action: np.ndarray):
        """Record decision for analysis"""
        
        decision_record = {
            'timestamp': time.time(),
            'tactical_state': self.current_tactical_state.value,
            'distance': features['distance'],
            'threat_level': features['threat_level'],
            'tactical_decision': tactical_decision['maneuver_type'],
            'final_action': final_action.tolist(),
            'locked': features['locked'],
            'missile_available': features['missile_available']
        }
        
        self.decision_history.append(decision_record)
        
        # Limit history size
        if len(self.decision_history) > 1000:
            self.decision_history.pop(0)
    
    def get_agent_analysis(self) -> Dict[str, Any]:
        """Get analysis of agent decision making"""
        
        if not self.decision_history:
            return {'status': 'no_data'}
        
        recent_decisions = self.decision_history[-100:]  # Last 100 decisions
        
        # Analyze decision patterns
        state_distribution = defaultdict(int)
        maneuver_distribution = defaultdict(int)
        firing_decisions = 0
        
        for decision in recent_decisions:
            state_distribution[decision['tactical_state']] += 1
            maneuver_distribution[decision['tactical_decision']] += 1
            if decision['final_action'][3] > 0.5:
                firing_decisions += 1
        
        return {
            'agent_type': self.agent_type,
            'skill_level': self.skill_level,
            'decisions_analyzed': len(recent_decisions),
            'state_distribution': dict(state_distribution),
            'maneuver_distribution': dict(maneuver_distribution),
            'firing_rate': firing_decisions / len(recent_decisions),
            'current_tactical_state': self.current_tactical_state.value,
            'reaction_time': self.reaction_time,
            'accuracy_factor': self.accuracy_factor
        }


class SpecializedRuleAgents:
    """Collection of specialized rule-based agents for different purposes"""
    
    @staticmethod
    def create_bvr_specialist() -> EnhancedRuleAgent:
        """Create agent specialized in BVR combat"""
        
        class BVRSpecialist(EnhancedRuleAgent):
            def __init__(self):
                super().__init__("expert", skill_level=0.9)
                # Override doctrine for BVR focus
                self.doctrine['bvr_engagement_range'] = (10000, 20000)
                self.doctrine['firing_discipline'] = 0.95
                self.doctrine['energy_management_priority'] = 0.9
        
        return BVRSpecialist()
    
    @staticmethod
    def create_wvr_specialist() -> EnhancedRuleAgent:
        """Create agent specialized in WVR dogfighting"""
        
        class WVRSpecialist(EnhancedRuleAgent):
            def __init__(self):
                super().__init__("ace", skill_level=0.95)
                # Override doctrine for WVR focus
                self.doctrine['wvr_engagement_range'] = (1000, 5000)
                self.doctrine['energy_management_priority'] = 0.95
                self.reaction_time = 1  # Very fast reactions
        
        return WVRSpecialist()
    
    @staticmethod
    def create_defensive_specialist() -> EnhancedRuleAgent:
        """Create agent specialized in defensive tactics"""
        
        class DefensiveSpecialist(EnhancedRuleAgent):
            def __init__(self):
                super().__init__("expert", skill_level=0.85)
                # Override doctrine for defensive focus
                self.doctrine['threat_reaction_threshold'] = 0.2  # Very sensitive
                self.doctrine['energy_management_priority'] = 0.95
        
        return DefensiveSpecialist()
    
    @staticmethod
    def create_aggressive_attacker() -> EnhancedRuleAgent:
        """Create aggressive attacking agent"""
        
        class AggressiveAttacker(EnhancedRuleAgent):
            def __init__(self):
                super().__init__("ace", skill_level=0.8)
                # Override doctrine for aggressive tactics
                self.doctrine['firing_discipline'] = 0.7  # More willing to shoot
                self.doctrine['threat_reaction_threshold'] = 0.8  # Less defensive
        
        return AggressiveAttacker()


class RuleAgentFactory:
    """Factory for creating different rule-based agents"""
    
    @staticmethod
    def create_agent(agent_type: str, **kwargs) -> EnhancedRuleAgent:
        """
        Create specific rule-based agent
        
        Args:
            agent_type: Type of agent to create
            **kwargs: Additional configuration
        
        Returns:
            Configured rule-based agent
        """
        
        if agent_type == "bvr_specialist":
            return SpecializedRuleAgents.create_bvr_specialist()
        elif agent_type == "wvr_specialist":
            return SpecializedRuleAgents.create_wvr_specialist()
        elif agent_type == "defensive_specialist":
            return SpecializedRuleAgents.create_defensive_specialist()
        elif agent_type == "aggressive_attacker":
            return SpecializedRuleAgents.create_aggressive_attacker()
        elif agent_type == "novice":
            return EnhancedRuleAgent("novice", skill_level=0.3)
        elif agent_type == "competent":
            return EnhancedRuleAgent("competent", skill_level=0.6)
        elif agent_type == "expert":
            return EnhancedRuleAgent("expert", skill_level=0.8)
        elif agent_type == "ace":
            return EnhancedRuleAgent("ace", skill_level=0.95)
        else:
            return EnhancedRuleAgent("competent", skill_level=0.6)
    
    @staticmethod
    def get_available_agents() -> Dict[str, Dict[str, Any]]:
        """Get information about available rule-based agents"""
        
        return {
            'bvr_specialist': {
                'description': 'Expert in Beyond Visual Range combat',
                'skill_level': 0.9,
                'specialization': 'Long-range missile engagement'
            },
            'wvr_specialist': {
                'description': 'Expert in Within Visual Range dogfighting',
                'skill_level': 0.95,
                'specialization': 'Close-range air combat maneuvering'
            },
            'defensive_specialist': {
                'description': 'Expert in defensive tactics and evasion',
                'skill_level': 0.85,
                'specialization': 'Threat evasion and survival'
            },
            'aggressive_attacker': {
                'description': 'Aggressive offensive tactics',
                'skill_level': 0.8,
                'specialization': 'Offensive engagement and pursuit'
            },
            'novice': {
                'description': 'Novice pilot with basic skills',
                'skill_level': 0.3,
                'specialization': 'Basic flight and simple tactics'
            },
            'competent': {
                'description': 'Competent pilot with solid fundamentals',
                'skill_level': 0.6,
                'specialization': 'Standard tactical procedures'
            },
            'expert': {
                'description': 'Expert pilot with advanced tactics',
                'skill_level': 0.8,
                'specialization': 'Advanced tactical operations'
            },
            'ace': {
                'description': 'Ace pilot with superior skills',
                'skill_level': 0.95,
                'specialization': 'Master-level air combat'
            }
        }


def create_rule_agent_comparison_suite() -> Dict[str, EnhancedRuleAgent]:
    """Create suite of rule agents for comprehensive comparison"""
    
    print("[RULE AGENT SUITE] Creating comprehensive rule agent comparison suite...")
    
    factory = RuleAgentFactory()
    available_agents = factory.get_available_agents()
    
    agent_suite = {}
    
    for agent_type, specs in available_agents.items():
        agent = factory.create_agent(agent_type)
        agent_suite[agent_type] = agent
        
        print(f"   ✅ {agent_type}: {specs['description']} (skill: {specs['skill_level']})")
    
    print(f"[RULE AGENT SUITE] Created {len(agent_suite)} specialized agents")
    return agent_suite


if __name__ == "__main__":
    print("Enhanced Rule-Based Agents for Combat Training")
    
    # Create agent suite
    agent_suite = create_rule_agent_comparison_suite()
    
    print(f"\nAgent suite ready with {len(agent_suite)} specialized agents")
    
    # Test an agent
    test_agent = agent_suite['expert']
    test_obs = np.random.randn(25)
    test_action = test_agent.get_action(test_obs)
    
    print(f"Test action: {test_action}")
    print("Enhanced rule agents ready for training and comparison!")
