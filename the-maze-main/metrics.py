"""
Metrics calculation for maze navigation evaluation
COSC 4368 - Spring 2026
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional


class MetricsCalculator:
    """Calculate and track all required metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.episodes = []
        self.successful_episodes = []
        self.turns_to_solution = []
        self.path_lengths = []
        self.deaths = []
        self.total_turns = []
        
        # Bonus metrics
        self.unique_cells_discovered = []
        self.total_cells_visited = []
        self.replanning_events = []
        self.learning_curve = []
    
    def add_episode(self, success: bool, turns: int, path_length: int, 
                    deaths: int, unique_cells: int, total_visits: int,
                    replanning_count: int = 0):
        """Add data from one episode"""
        self.episodes.append(len(self.episodes) + 1)
        self.successful_episodes.append(1 if success else 0)
        self.turns_to_solution.append(turns if success else None)
        self.path_lengths.append(path_length if success else None)
        self.deaths.append(deaths)
        self.total_turns.append(turns)
        
        self.unique_cells_discovered.append(unique_cells)
        self.total_cells_visited.append(total_visits)
        self.replanning_events.append(replanning_count)
    
    def calculate_success_rate(self) -> float:
        """Primary metric: success_rate = successful_episodes / total_episodes"""
        if not self.successful_episodes:
            return 0.0
        return sum(self.successful_episodes) / len(self.successful_episodes)
    
    def calculate_avg_path_length(self) -> float:
        """Primary metric: average path length (successful episodes only)"""
        valid_lengths = [l for l in self.path_lengths if l is not None]
        if not valid_lengths:
            return 0.0
        return np.mean(valid_lengths)
    
    def calculate_avg_turns(self) -> float:
        """Primary metric: average turns to solution (successful episodes only)"""
        valid_turns = [t for t in self.turns_to_solution if t is not None]
        if not valid_turns:
            return 0.0
        return np.mean(valid_turns)
    
    def calculate_death_rate(self) -> float:
        """Primary metric: death_rate = total_deaths / total_turns"""
        total_deaths = sum(self.deaths)
        total_turns = sum(self.total_turns)
        if total_turns == 0:
            return 0.0
        return total_deaths / total_turns
    
    def calculate_exploration_efficiency(self) -> float:
        """Bonus metric: unique_cells_discovered / total_cells_visited"""
        valid_ratios = []
        for unique, total in zip(self.unique_cells_discovered, self.total_cells_visited):
            if total > 0:
                valid_ratios.append(unique / total)
        if not valid_ratios:
            return 0.0
        return np.mean(valid_ratios)
    
    def calculate_map_completeness(self, total_navigable_cells: int = 4096) -> float:
        """Bonus metric: known_cells / total_navigable_cells"""
        if not self.unique_cells_discovered:
            return 0.0
        avg_unique = np.mean(self.unique_cells_discovered)
        return avg_unique / total_navigable_cells
    
    def calculate_replanning_efficiency(self) -> float:
        """Bonus metric: average replanning events per episode (lower is better)"""
        if not self.replanning_events:
            return 0.0
        return np.mean(self.replanning_events)
    
    def get_all_metrics(self, total_navigable_cells: int = 4096) -> Dict:
        """Return all metrics as a dictionary"""
        return {
            # Primary metrics (required)
            'success_rate': self.calculate_success_rate(),
            'avg_path_length': self.calculate_avg_path_length(),
            'avg_turns_to_solution': self.calculate_avg_turns(),
            'death_rate': self.calculate_death_rate(),
            
            # Bonus metrics
            'exploration_efficiency': self.calculate_exploration_efficiency(),
            'map_completeness': self.calculate_map_completeness(total_navigable_cells),
            'replanning_efficiency': self.calculate_replanning_efficiency(),
        }
    
    def print_summary(self):
        """Print formatted metrics summary"""
        metrics = self.get_all_metrics()
        
        print("\n" + "=" * 50)
        print("METRICS SUMMARY")
        print("=" * 50)
        print(f"Success Rate:              {metrics['success_rate']*100:.1f}%")
        print(f"Average Path Length:       {metrics['avg_path_length']:.1f} cells")
        print(f"Average Turns to Solution: {metrics['avg_turns_to_solution']:.1f} turns")
        print(f"Death Rate:                {metrics['death_rate']:.4f}")
        print("-" * 50)
        print(f"Exploration Efficiency:    {metrics['exploration_efficiency']:.3f}")
        print(f"Map Completeness:          {metrics['map_completeness']*100:.1f}%")
        print(f"Replanning Efficiency:     {metrics['replanning_efficiency']:.1f} events/ep")
        print("=" * 50)
    
    def save_to_json(self, filename: str):
        """Save metrics to JSON file"""
        metrics = self.get_all_metrics()
        metrics['raw_data'] = {
            'episodes': self.episodes,
            'successful_episodes': self.successful_episodes,
            'turns_to_solution': self.turns_to_solution,
            'path_lengths': self.path_lengths,
            'deaths': self.deaths,
            'total_turns': self.total_turns,
            'unique_cells_discovered': self.unique_cells_discovered,
            'total_cells_visited': self.total_cells_visited,
            'replanning_events': self.replanning_events
        }
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {filename}")


class LearningCurveAnalyzer:
    """Analyze learning progress over time"""
    
    @staticmethod
    def calculate_convergence_episode(success_rates: List[float], 
                                       window_size: int = 50,
                                       threshold: float = 0.9) -> int:
        """
        Find episode where agent converged (rolling average above threshold)
        """
        if len(success_rates) < window_size:
            return len(success_rates)
        
        for i in range(window_size, len(success_rates)):
            rolling_avg = np.mean(success_rates[i-window_size:i])
            if rolling_avg >= threshold:
                return i
        return len(success_rates)
    
    @staticmethod
    def calculate_sample_efficiency(num_episodes_to_converge: int, 
                                    success_rate_at_convergence: float) -> float:
        """Sample efficiency = success_rate / episodes_to_converge"""
        if num_episodes_to_converge == 0:
            return 0.0
        return success_rate_at_convergence / num_episodes_to_converge


def compute_all_metrics_from_episodes(episode_data: List[Dict]) -> Dict:
    """
    Compute all metrics from a list of episode dictionaries
    
    Args:
        episode_data: List of dicts with keys:
            - success (bool)
            - turns (int)
            - path_length (int)
            - deaths (int)
            - unique_cells (int)
            - total_visits (int)
    
    Returns:
        Dictionary with all metrics
    """
    calc = MetricsCalculator()
    
    for ep in episode_data:
        calc.add_episode(
            success=ep.get('success', False),
            turns=ep.get('turns', 0),
            path_length=ep.get('path_length', 0),
            deaths=ep.get('deaths', 0),
            unique_cells=ep.get('unique_cells', 0),
            total_visits=ep.get('total_visits', 0),
            replanning_count=ep.get('replanning_count', 0)
        )
    
    return calc.get_all_metrics()


if __name__ == "__main__":
    # Demo usage
    calc = MetricsCalculator()
    
    # Simulate some episodes
    calc.add_episode(success=True, turns=500, path_length=400, deaths=2,
                     unique_cells=800, total_visits=1200)
    calc.add_episode(success=True, turns=450, path_length=380, deaths=1,
                     unique_cells=750, total_visits=1100)
    calc.add_episode(success=False, turns=10000, path_length=0, deaths=5,
                     unique_cells=600, total_visits=800)
    calc.add_episode(success=True, turns=420, path_length=350, deaths=0,
                     unique_cells=820, total_visits=1150)
    calc.add_episode(success=True, turns=480, path_length=390, deaths=1,
                     unique_cells=790, total_visits=1180)
    
    calc.print_summary()
    calc.save_to_json('demo_metrics.json')