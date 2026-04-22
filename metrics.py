"""
Metrics Calculator — COSC 4368 Spring 2026
Computes all required + bonus metrics for maze navigation.
"""

import json
import numpy as np
from typing import List, Dict


class MetricsCalculator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.episodes = []

    def add_episode(self, success: bool, turns: int, path_length: int,
                    deaths: int, unique_cells: int, total_visits: int,
                    replanning_count: int = 0):
        self.episodes.append({
            'success': success,
            'turns': turns,
            'path_length': path_length,
            'deaths': deaths,
            'unique_cells': unique_cells,
            'total_visits': total_visits,
            'replanning_count': replanning_count,
        })

    # ── Required Metrics ──────────────────────────────────────────────────────
    def success_rate(self) -> float:
        if not self.episodes: return 0.0
        return sum(1 for e in self.episodes if e['success']) / len(self.episodes)

    def avg_path_length(self) -> float:
        ok = [e['path_length'] for e in self.episodes if e['success']]
        return float(np.mean(ok)) if ok else 0.0

    def avg_turns_to_solution(self) -> float:
        ok = [e['turns'] for e in self.episodes if e['success']]
        return float(np.mean(ok)) if ok else 0.0

    def death_rate(self) -> float:
        total_deaths = sum(e['deaths'] for e in self.episodes)
        total_turns  = sum(e['turns'] for e in self.episodes)
        return total_deaths / total_turns if total_turns > 0 else 0.0

    # ── Bonus Metrics ─────────────────────────────────────────────────────────
    def exploration_efficiency(self) -> float:
        ratios = [e['unique_cells'] / e['total_visits']
                  for e in self.episodes if e['total_visits'] > 0]
        return float(np.mean(ratios)) if ratios else 0.0

    def map_completeness(self, total_navigable: int = 4096) -> float:
        if not self.episodes: return 0.0
        return float(np.mean([e['unique_cells'] for e in self.episodes])) / total_navigable

    def replanning_efficiency(self) -> float:
        if not self.episodes: return 0.0
        return float(np.mean([e['replanning_count'] for e in self.episodes]))

    def learning_efficiency(self) -> List[float]:
        """Rolling success rate to show learning curve."""
        window = max(5, len(self.episodes) // 10)
        curve = []
        for i in range(len(self.episodes)):
            start = max(0, i - window + 1)
            segment = self.episodes[start:i+1]
            rate = sum(1 for e in segment if e['success']) / len(segment)
            curve.append(rate)
        return curve

    # ── Summary ───────────────────────────────────────────────────────────────
    def get_all(self, total_navigable: int = 4096) -> Dict:
        return {
            'success_rate': self.success_rate(),
            'avg_path_length': self.avg_path_length(),
            'avg_turns_to_solution': self.avg_turns_to_solution(),
            'death_rate': self.death_rate(),
            'exploration_efficiency': self.exploration_efficiency(),
            'map_completeness': self.map_completeness(total_navigable),
            'replanning_efficiency': self.replanning_efficiency(),
            'learning_curve': self.learning_efficiency(),
            'total_episodes': len(self.episodes),
        }

    def print_summary(self, label=""):
        m = self.get_all()
        print(f"\n{'='*55}")
        print(f"  METRICS — {label}")
        print(f"{'='*55}")
        print(f"  Success Rate:              {m['success_rate']*100:6.1f}%")
        print(f"  Average Path Length:        {m['avg_path_length']:6.1f} cells")
        print(f"  Average Turns to Solution:  {m['avg_turns_to_solution']:6.1f} turns")
        print(f"  Death Rate:                 {m['death_rate']:8.4f}")
        print(f"  {'─'*50}")
        print(f"  Exploration Efficiency:     {m['exploration_efficiency']:8.3f}")
        print(f"  Map Completeness:           {m['map_completeness']*100:6.1f}%")
        print(f"  Replanning Efficiency:      {m['replanning_efficiency']:6.1f} events/ep")
        print(f"  Total Episodes:             {m['total_episodes']}")
        print(f"{'='*55}\n")

    def save_json(self, filename: str):
        m = self.get_all()
        m['episode_data'] = self.episodes
        with open(filename, 'w') as f:
            json.dump(m, f, indent=2, default=str)
        print(f"  Saved metrics → {filename}")
