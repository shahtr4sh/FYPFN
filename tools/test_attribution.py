import random
import numpy as np
import sys
import os
# ensure project root is on sys.path so imports work when running this script
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from simulator import FakeNewsSimulator

def run_test(num_agents=30, rounds=3):
    modes = ['weighted', 'random', 'first']
    results = {}
    for mode in modes:
        # set seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        sim = FakeNewsSimulator(num_agents)
        sim.attribution_mode = mode
        # seed two initial sharers
        sim.seed_initial_state(is_scam=False)
        # run a few rounds
        for r in range(rounds):
            sim.simulate_fake_news_round(juice_factor=0.5, topic_weight=0.5, topic_category=None)
        results[mode] = sim.transmission_history if hasattr(sim, 'transmission_history') else []
    return results

if __name__ == '__main__':
    res = run_test()
    for mode, hist in res.items():
        print(f"--- Mode: {mode} ---")
        for i, edges in enumerate(hist, start=1):
            print(f"Round {i}: {edges}")
        print()
