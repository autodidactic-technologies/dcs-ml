# Specialized Harfang Environment Variants for Enhanced Training Diversity
import numpy as np
import math
import random
import time
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from env.mock_harfang_env import MockHarfangEnhancedEnv


class FlightPattern(Enum):
    """Different enemy flight patterns for training diversity"""
    STRAIGHT_LINE = "straight_line"
    SERPENTINE = "serpentine"
    CIRCULAR = "circular"
    ZIGZAG = "zigzag"
    DEFENSIVE_SPIRAL = "defensive_spiral"
    AGGRESSIVE_INTERCEPT = "aggressive_intercept"


class EnemySkillLevel(Enum):
    """Enemy AI skill levels"""
    NOVICE = "novice"      # 0.2 skill
    BASIC = "basic"        # 0.4 skill
    COMPETENT = "competent"  # 0.6 skill
    EXPERT = "expert"      # 0.8 skill
    ACE = "ace"           # 1.0 skill


class HarfangTacticalEnv(MockHarfangEnhancedEnv):
    """
    Advanced tactical environment with realistic air combat doctrine
    and intelligent enemy AI that follows proper BVR/WVR tactics
    """
    
    def __init__(self, max_episode_steps: int = 2000, enemy_skill: EnemySkillLevel = EnemySkillLevel.COMPETENT):
        """
        Initialize tactical environment
        
        Args:
            max_episode_steps: Maximum episode length
            enemy_skill: Enemy AI skill level
        """
        super().__init__(max_episode_steps)
        
        self.enemy_skill = enemy_skill
        self.skill_value = self._get_skill_value(enemy_skill)
        
        # Tactical state variables
        self.combat_phase = "BVR"  # BVR, MERGE, WVR, DEFENSIVE, OFFENSIVE
        self.last_maneuver = "NONE"
        self.maneuver_timer = 0
        self.evasion_timer = 0
        self.offensive_timer = 0
        
        # Advanced tactical parameters
        self.BVR_RANGE = 15000     # Beyond Visual Range
        self.MERGE_RANGE = 8000    # Merge phase
        self.WVR_RANGE = 3000      # Within Visual Range
        self.DEFENSIVE_RANGE = 1500 # Defensive maneuvering
        
        # Enemy AI parameters
        self.enemy_decision_timer = 0
        self.enemy_target_locked_on_us = False
        self.enemy_missile_fired = False
        self.enemy_last_maneuver = "NONE"
        
        print(f"[TACTICAL ENV] Initialized with {enemy_skill.value} enemy (skill: {self.skill_value:.1f})")
    
    def _get_skill_value(self, skill_level: EnemySkillLevel) -> float:
        """Convert skill level to numeric value"""
        skill_mapping = {
            EnemySkillLevel.NOVICE: 0.2,
            EnemySkillLevel.BASIC: 0.4,
            EnemySkillLevel.COMPETENT: 0.6,
            EnemySkillLevel.EXPERT: 0.8,
            EnemySkillLevel.ACE: 1.0
        }
        return skill_mapping[skill_level]
    
    def _update_enemy(self):
        """Enhanced enemy AI with tactical doctrine"""
        
        distance = np.linalg.norm(self.enemy_pos - self.ego_pos)
        
        # Update combat phase
        self._update_combat_phase(distance)
        
        # Enemy decision making based on skill and phase
        self.enemy_decision_timer += 1
        
        if self.enemy_decision_timer >= (10 - int(self.skill_value * 5)):  # Faster decisions for higher skill
            self._make_enemy_tactical_decision(distance)
            self.enemy_decision_timer = 0
        
        # Execute current maneuver
        self._execute_enemy_maneuver(distance)
        
        # Update enemy position
        super()._update_enemy()
    
    def _update_combat_phase(self, distance: float):
        """Update combat phase based on distance and tactical situation"""
        
        if distance > self.BVR_RANGE:
            self.combat_phase = "BVR"
        elif distance > self.MERGE_RANGE:
            self.combat_phase = "MERGE"
        elif distance > self.WVR_RANGE:
            self.combat_phase = "WVR"
        else:
            self.combat_phase = "KNIFE_FIGHT"
        
        # Override based on tactical situation
        if self.threat_level > 0.7:
            self.combat_phase = "DEFENSIVE"
        elif self.locked and distance < self.WVR_RANGE:
            self.combat_phase = "OFFENSIVE"
    
    def _make_enemy_tactical_decision(self, distance: float):
        """Make tactical decision based on combat doctrine"""
        
        # Decision probability based on skill
        decision_quality = self.skill_value + random.uniform(-0.1, 0.1)
        
        if self.combat_phase == "BVR":
            # BVR tactics: Maintain distance, use radar missiles
            if decision_quality > 0.6:
                if distance < self.BVR_RANGE * 0.8:
                    self.enemy_last_maneuver = "EXTEND"  # Create separation
                elif not self.enemy_target_locked_on_us and random.random() < 0.3:
                    self.enemy_last_maneuver = "LOCK_ATTEMPT"
                else:
                    self.enemy_last_maneuver = "MAINTAIN_RANGE"
        
        elif self.combat_phase == "MERGE":
            # Merge tactics: Positioning for advantage
            if decision_quality > 0.5:
                if random.random() < 0.4:
                    self.enemy_last_maneuver = "CRANK"  # Maintain missile range
                else:
                    self.enemy_last_maneuver = "BEAM"   # Beam maneuver
        
        elif self.combat_phase == "WVR":
            # WVR tactics: Energy fighting and maneuvering
            if decision_quality > 0.4:
                if self.ego_pos[2] > self.enemy_pos[2]:  # We have altitude advantage
                    self.enemy_last_maneuver = "DEFENSIVE_SPIRAL"
                else:
                    self.enemy_last_maneuver = "AGGRESSIVE_TURN"
        
        elif self.combat_phase == "DEFENSIVE":
            # Defensive tactics: Evasion and countermeasures
            if decision_quality > 0.3:
                self.enemy_last_maneuver = "NOTCH"  # Notch maneuver
            else:
                self.enemy_last_maneuver = "CHAFF_AND_TURN"
    
    def _execute_enemy_maneuver(self, distance: float):
        """Execute enemy tactical maneuver"""
        
        maneuver = self.enemy_last_maneuver
        
        if maneuver == "EXTEND":
            # Extend away to create separation
            direction = (self.enemy_pos - self.ego_pos) / distance
            self.enemy_pos += direction * 200.0  # Fast extension
        
        elif maneuver == "CRANK":
            # Crank maneuver - maintain range while maneuvering
            to_ego = self.ego_pos - self.enemy_pos
            perpendicular = np.array([-to_ego[1], to_ego[0], 0])
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
            self.enemy_pos += perpendicular * 150.0
        
        elif maneuver == "DEFENSIVE_SPIRAL":
            # Defensive spiral maneuver
            self.maneuver_timer += 1
            angle = self.maneuver_timer * 0.2
            spiral_radius = 2000
            self.enemy_pos[0] += spiral_radius * math.cos(angle) * 0.01
            self.enemy_pos[1] += spiral_radius * math.sin(angle) * 0.01
            self.enemy_pos[2] -= 50  # Descending spiral
        
        elif maneuver == "NOTCH":
            # 90-degree turn to notch incoming missile
            to_ego = self.ego_pos - self.enemy_pos
            perpendicular = np.array([-to_ego[1], to_ego[0], 0])
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
            self.enemy_pos += perpendicular * 180.0
        
        else:
            # Default behavior
            super()._update_enemy()


class HarfangSerpentineEnv(MockHarfangEnhancedEnv):
    """Environment with serpentine enemy flight pattern"""
    
    def __init__(self, max_episode_steps: int = 2000):
        super().__init__(max_episode_steps)
        self.serpentine_timer = 0
        self.serpentine_direction = 1
        print("[SERPENTINE ENV] Serpentine enemy pattern initialized")
    
    def _update_enemy(self):
        """Update enemy with serpentine pattern"""
        
        self.serpentine_timer += 1
        
        # Serpentine pattern - S-shaped flight path
        base_velocity = np.array([-150.0, 0.0, 0.0])  # Base westward movement
        
        # Add serpentine oscillation
        oscillation_amplitude = 100.0
        oscillation_frequency = 0.05
        
        serpentine_offset = oscillation_amplitude * math.sin(self.serpentine_timer * oscillation_frequency)
        
        # Change direction periodically
        if self.serpentine_timer % 200 == 0:
            self.serpentine_direction *= -1
        
        self.enemy_velocity = base_velocity + np.array([0, serpentine_offset * self.serpentine_direction, 0])
        
        # Update position
        self.enemy_pos += self.enemy_velocity * 1.0  # 1 second timestep
        
        # Add some randomness for realism
        noise = np.random.uniform(-10, 10, 3)
        self.enemy_pos += noise


class HarfangCircularEnv(MockHarfangEnhancedEnv):
    """Environment with circular/orbital enemy flight pattern"""
    
    def __init__(self, max_episode_steps: int = 2000, orbit_radius: float = 8000):
        super().__init__(max_episode_steps)
        self.orbit_radius = orbit_radius
        self.orbit_angle = 0.0
        self.orbit_center = np.array([5000.0, 0.0, 5000.0])  # Center of orbit
        print(f"[CIRCULAR ENV] Circular enemy pattern initialized (radius: {orbit_radius}m)")
    
    def _update_enemy(self):
        """Update enemy with circular orbital pattern"""
        
        # Update orbit angle
        angular_velocity = 0.02  # radians per step
        self.orbit_angle += angular_velocity
        
        # Calculate new position on orbit
        self.enemy_pos = self.orbit_center + np.array([
            self.orbit_radius * math.cos(self.orbit_angle),
            self.orbit_radius * math.sin(self.orbit_angle),
            0  # Maintain altitude
        ])
        
        # Add some altitude variation
        altitude_variation = 500 * math.sin(self.orbit_angle * 2)
        self.enemy_pos[2] += altitude_variation
        
        # Update velocity for realistic movement
        self.enemy_velocity = np.array([
            -self.orbit_radius * angular_velocity * math.sin(self.orbit_angle),
            self.orbit_radius * angular_velocity * math.cos(self.orbit_angle),
            altitude_variation * angular_velocity * 2
        ])


class HarfangDoctrineEnemyEnv(MockHarfangEnhancedEnv):
    """Environment with enemy AI following real air combat doctrine"""
    
    def __init__(self, max_episode_steps: int = 2000, enemy_doctrine: str = "aggressive"):
        super().__init__(max_episode_steps)
        
        self.enemy_doctrine = enemy_doctrine  # aggressive, defensive, balanced
        self.doctrine_parameters = self._load_doctrine_parameters(enemy_doctrine)
        
        # Doctrine-specific state
        self.enemy_energy_state = "HIGH"
        self.enemy_tactical_priority = "BVR_FIRST_SHOT"
        self.enemy_threat_assessment = 0.0
        self.enemy_decision_history = []
        
        # Initialize combat_phase for compatibility
        self.combat_phase = "BVR"
        
        print(f"[DOCTRINE ENV] Enemy with {enemy_doctrine} doctrine initialized")
    
    def _load_doctrine_parameters(self, doctrine: str) -> Dict[str, float]:
        """Load doctrine-specific parameters"""
        
        doctrines = {
            'aggressive': {
                'bvr_engagement_range': 12000,
                'wvr_transition_range': 4000,
                'energy_priority': 0.8,
                'offensive_bias': 0.9,
                'risk_tolerance': 0.8
            },
            'defensive': {
                'bvr_engagement_range': 18000,
                'wvr_transition_range': 6000,
                'energy_priority': 0.9,
                'offensive_bias': 0.3,
                'risk_tolerance': 0.3
            },
            'balanced': {
                'bvr_engagement_range': 15000,
                'wvr_transition_range': 5000,
                'energy_priority': 0.7,
                'offensive_bias': 0.6,
                'risk_tolerance': 0.6
            }
        }
        
        return doctrines.get(doctrine, doctrines['balanced'])
    
    def _update_enemy(self):
        """Update enemy using combat doctrine"""
        
        distance = np.linalg.norm(self.enemy_pos - self.ego_pos)
        
        # Assess tactical situation
        self._assess_enemy_tactical_situation(distance)
        
        # Make doctrine-based decision
        maneuver = self._select_doctrine_maneuver(distance)
        
        # Execute maneuver
        self._execute_doctrine_maneuver(maneuver, distance)
        
        # Update decision history
        self.enemy_decision_history.append({
            'step': self.current_step,
            'distance': distance,
            'maneuver': maneuver,
            'combat_phase': self.combat_phase,
            'energy_state': self.enemy_energy_state
        })
        
        # Limit history size
        if len(self.enemy_decision_history) > 100:
            self.enemy_decision_history.pop(0)
    
    def _assess_enemy_tactical_situation(self, distance: float):
        """Assess tactical situation from enemy perspective"""
        
        # Update enemy energy state
        enemy_altitude = self.enemy_pos[2]
        if enemy_altitude > 8000:
            self.enemy_energy_state = "HIGH"
        elif enemy_altitude < 3000:
            self.enemy_energy_state = "LOW"
        else:
            self.enemy_energy_state = "MEDIUM"
        
        # Update threat assessment
        if self.locked:  # We have lock on enemy
            self.enemy_threat_assessment = 0.8
        elif distance < 5000:
            self.enemy_threat_assessment = 0.6
        elif distance < 10000:
            self.enemy_threat_assessment = 0.4
        else:
            self.enemy_threat_assessment = 0.2
        
        # Update tactical priority
        if distance > self.doctrine_parameters['bvr_engagement_range']:
            self.enemy_tactical_priority = "BVR_POSITIONING"
        elif distance > self.doctrine_parameters['wvr_transition_range']:
            self.enemy_tactical_priority = "BVR_ENGAGEMENT"
        elif self.enemy_threat_assessment > 0.7:
            self.enemy_tactical_priority = "DEFENSIVE_MANEUVERS"
        else:
            self.enemy_tactical_priority = "WVR_ENGAGEMENT"
    
    def _select_doctrine_maneuver(self, distance: float) -> str:
        """Select maneuver based on doctrine and tactical situation"""
        
        doctrine = self.doctrine_parameters
        
        # Doctrine-based decision making
        if self.enemy_tactical_priority == "BVR_POSITIONING":
            if self.enemy_doctrine == "aggressive":
                return "AGGRESSIVE_APPROACH"
            elif self.enemy_doctrine == "defensive":
                return "MAINTAIN_STANDOFF"
            else:
                return "CONTROLLED_APPROACH"
        
        elif self.enemy_tactical_priority == "BVR_ENGAGEMENT":
            if doctrine['offensive_bias'] > 0.7:
                return "CRANK_FOR_SHOT"
            else:
                return "BEAM_MANEUVER"
        
        elif self.enemy_tactical_priority == "DEFENSIVE_MANEUVERS":
            if self.enemy_energy_state == "HIGH":
                return "DEFENSIVE_SPIRAL"
            else:
                return "NOTCH_AND_EXTEND"
        
        elif self.enemy_tactical_priority == "WVR_ENGAGEMENT":
            if self.enemy_energy_state == "HIGH":
                return "AGGRESSIVE_TURN"
            else:
                return "ENERGY_CONSERVATION"
        
        return "MAINTAIN_COURSE"
    
    def _execute_doctrine_maneuver(self, maneuver: str, distance: float):
        """Execute specific tactical maneuver"""
        
        if maneuver == "AGGRESSIVE_APPROACH":
            # Direct approach with slight offset
            to_ego = self.ego_pos - self.enemy_pos
            approach_vector = to_ego / np.linalg.norm(to_ego)
            # Add slight offset for tactical approach
            offset = np.array([approach_vector[1], -approach_vector[0], 0]) * 0.2
            self.enemy_pos += (approach_vector + offset) * 250.0
        
        elif maneuver == "MAINTAIN_STANDOFF":
            # Maintain distance while maneuvering
            to_ego = self.ego_pos - self.enemy_pos
            if distance < self.BVR_RANGE:
                # Move away
                self.enemy_pos -= (to_ego / np.linalg.norm(to_ego)) * 200.0
            elif distance > self.BVR_RANGE * 1.5:
                # Move closer
                self.enemy_pos += (to_ego / np.linalg.norm(to_ego)) * 150.0
        
        elif maneuver == "CRANK_FOR_SHOT":
            # Crank maneuver to maintain missile range
            to_ego = self.ego_pos - self.enemy_pos
            perpendicular = np.array([-to_ego[1], to_ego[0], 0])
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
            self.enemy_pos += perpendicular * 180.0
        
        elif maneuver == "DEFENSIVE_SPIRAL":
            # Defensive spiral with altitude loss
            self.maneuver_timer += 1
            angle = self.maneuver_timer * 0.3
            spiral_radius = 1500
            self.enemy_pos[0] += spiral_radius * math.cos(angle) * 0.01
            self.enemy_pos[1] += spiral_radius * math.sin(angle) * 0.01
            self.enemy_pos[2] -= 100  # Lose altitude
        
        elif maneuver == "NOTCH_AND_EXTEND":
            # 90-degree turn and extend
            to_ego = self.ego_pos - self.enemy_pos
            perpendicular = np.array([-to_ego[1], to_ego[0], 0])
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
            self.enemy_pos += perpendicular * 200.0
        
        else:
            # Default maneuver
            super()._update_enemy()


class HarfangVariantFactory:
    """Factory for creating different Harfang environment variants"""
    
    @staticmethod
    def create_environment(variant_type: str, **kwargs) -> MockHarfangEnhancedEnv:
        """
        Create specific environment variant
        
        Args:
            variant_type: Type of environment variant
            **kwargs: Additional configuration parameters
        
        Returns:
            Configured environment variant
        """
        
        max_steps = kwargs.get('max_episode_steps', 2000)
        
        if variant_type == "tactical":
            enemy_skill = kwargs.get('enemy_skill', EnemySkillLevel.COMPETENT)
            return HarfangTacticalEnv(max_steps, enemy_skill)
        
        elif variant_type == "serpentine":
            return HarfangSerpentineEnv(max_steps)
        
        elif variant_type == "circular":
            orbit_radius = kwargs.get('orbit_radius', 8000)
            return HarfangCircularEnv(max_steps, orbit_radius)
        
        elif variant_type == "doctrine":
            doctrine = kwargs.get('enemy_doctrine', 'aggressive')
            return HarfangDoctrineEnemyEnv(max_steps, doctrine)
        
        else:
            # Default enhanced environment
            return MockHarfangEnhancedEnv(max_steps)
    
    @staticmethod
    def get_available_variants() -> Dict[str, Dict[str, Any]]:
        """Get information about available environment variants"""
        
        return {
            'tactical': {
                'description': 'Advanced tactical environment with skilled enemy AI',
                'parameters': ['enemy_skill'],
                'difficulty': 'high',
                'focus': 'realistic_combat_doctrine'
            },
            'serpentine': {
                'description': 'Enemy follows serpentine flight pattern',
                'parameters': [],
                'difficulty': 'medium',
                'focus': 'tracking_moving_targets'
            },
            'circular': {
                'description': 'Enemy follows circular orbital pattern',
                'parameters': ['orbit_radius'],
                'difficulty': 'medium',
                'focus': 'geometry_management'
            },
            'doctrine': {
                'description': 'Enemy follows specific combat doctrine',
                'parameters': ['enemy_doctrine'],
                'difficulty': 'high',
                'focus': 'doctrine_vs_doctrine_combat'
            }
        }


class EnvironmentVariantManager:
    """
    Manager for training across multiple environment variants
    to improve agent robustness and tactical adaptability
    """
    
    def __init__(self, variants: List[str] = None):
        """
        Initialize environment variant manager
        
        Args:
            variants: List of environment variants to use
        """
        if variants is None:
            variants = ['tactical', 'serpentine', 'circular', 'doctrine']
        
        self.variants = variants
        self.variant_configs = HarfangVariantFactory.get_available_variants()
        self.current_variant = 0
        self.variant_performance = {variant: [] for variant in variants}
        
        print(f"[VARIANT MANAGER] Initialized with {len(variants)} variants")
        for variant in variants:
            config = self.variant_configs[variant]
            print(f"   • {variant}: {config['description']} (difficulty: {config['difficulty']})")
    
    def get_next_environment(self, **kwargs) -> Tuple[MockHarfangEnhancedEnv, str]:
        """
        Get next environment variant for training
        
        Returns:
            (environment, variant_name)
        """
        
        variant_name = self.variants[self.current_variant]
        
        # Create environment with variant-specific parameters
        if variant_name == 'tactical':
            # Vary enemy skill
            skill_levels = list(EnemySkillLevel)
            enemy_skill = random.choice(skill_levels)
            env = HarfangVariantFactory.create_environment(variant_name, enemy_skill=enemy_skill, **kwargs)
        
        elif variant_name == 'circular':
            # Vary orbit radius
            orbit_radius = random.uniform(5000, 12000)
            env = HarfangVariantFactory.create_environment(variant_name, orbit_radius=orbit_radius, **kwargs)
        
        elif variant_name == 'doctrine':
            # Vary enemy doctrine
            doctrine = random.choice(['aggressive', 'defensive', 'balanced'])
            env = HarfangVariantFactory.create_environment(variant_name, enemy_doctrine=doctrine, **kwargs)
        
        else:
            env = HarfangVariantFactory.create_environment(variant_name, **kwargs)
        
        # Cycle to next variant
        self.current_variant = (self.current_variant + 1) % len(self.variants)
        
        return env, variant_name
    
    def record_variant_performance(self, variant_name: str, performance_metrics: Dict[str, float]):
        """Record performance on specific variant"""
        
        if variant_name in self.variant_performance:
            self.variant_performance[variant_name].append(performance_metrics)
    
    def get_variant_analysis(self) -> Dict[str, Any]:
        """Analyze performance across variants"""
        
        analysis = {}
        
        for variant, performance_list in self.variant_performance.items():
            if performance_list:
                success_rates = [p.get('success', False) for p in performance_list]
                avg_scores = [p.get('score', 0.0) for p in performance_list]
                
                analysis[variant] = {
                    'episodes': len(performance_list),
                    'success_rate': np.mean(success_rates),
                    'average_score': np.mean(avg_scores),
                    'difficulty': self.variant_configs[variant]['difficulty'],
                    'focus_area': self.variant_configs[variant]['focus']
                }
        
        return analysis


def integrate_variants_with_training(base_training_script: str = "enhanced_harfang_rl_llm.py"):
    """
    Integration function to add environment variants to main training script
    
    Args:
        base_training_script: Path to base training script
    
    Returns:
        Integration instructions
    """
    
    integration_code = '''
# Add to enhanced_harfang_rl_llm.py:

from environments.specialized_harfang_variants import (
    EnvironmentVariantManager, HarfangVariantFactory,
    EnemySkillLevel, FlightPattern
)

# In argument parser, add:
parser.add_argument('--env_variant', type=str, 
                   choices=['standard', 'tactical', 'serpentine', 'circular', 'doctrine', 'mixed'],
                   default='standard', help='Environment variant for training')
parser.add_argument('--enemy_skill', type=str,
                   choices=['novice', 'basic', 'competent', 'expert', 'ace'],
                   default='competent', help='Enemy AI skill level')

# In environment setup, replace:
if args.env_variant != 'standard':
    if args.env_variant == 'mixed':
        # Use variant manager for diverse training
        variant_manager = EnvironmentVariantManager()
        env, variant_name = variant_manager.get_next_environment(max_episode_steps=args.max_episode_steps)
        print(f"[ENV] Using mixed variants (current: {variant_name})")
    else:
        # Use specific variant
        env = HarfangVariantFactory.create_environment(
            args.env_variant,
            max_episode_steps=args.max_episode_steps,
            enemy_skill=EnemySkillLevel(args.enemy_skill)
        )
        print(f"[ENV] Using {args.env_variant} variant with {args.enemy_skill} enemy")
'''
    
    print("[INTEGRATION] Environment variant integration code generated")
    print("Add this code to enhanced_harfang_rl_llm.py for variant support")
    
    return integration_code


if __name__ == "__main__":
    print("Specialized Harfang Environment Variants")
    
    # Test variant creation
    factory = HarfangVariantFactory()
    
    print("\nTesting environment variants:")
    
    # Test each variant
    variants = ['tactical', 'serpentine', 'circular', 'doctrine']
    
    for variant in variants:
        try:
            env = factory.create_environment(variant, max_episode_steps=100)
            obs, info = env.reset()
            print(f"✅ {variant}: obs shape {obs.shape}")
            env.close()
        except Exception as e:
            print(f"❌ {variant}: {e}")
    
    print("\nEnvironment variants ready for integration!")
