"""Run a deterministic 24-round demo comparing ABM and PBM believer counts.
Saves results to stdout for quick inspection.
"""
import random
import numpy as np
import sys
import os

# Ensure repo root is on path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from simulator import FakeNewsSimulator
from pbm_simulator import PopulationSimulator

# Deterministic seeds
random.seed(12345)
np.random.seed(12345)

NUM_AGENTS = 50

# Create ABM with random agents (to emulate a run)
abm = FakeNewsSimulator(NUM_AGENTS)
abm.seed_initial_state()

# Create PBM
pbm = PopulationSimulator(NUM_AGENTS)
pbm.adjust_rates(topic_weight=1.0, juice_factor=0.5, intervention=False)

# Collect time series
abm_counts = [sum(1 for v in abm.agent_states.values() if v.get('shared'))]
pbm_counts = [pbm.believers]

# Run 24 rounds, enable intervention at round 6 for test
for r in range(1, 24+1):
    intervention = (r >= 6)
    # run ABM round
    abm.simulate_fake_news_round(juice_factor=0.5, topic_weight=1.0, topic_category=None, intervention=intervention)
    abm_count = sum(1 for v in abm.agent_states.values() if v.get('shared'))
    abm_counts.append(abm_count)

    # run PBM step
    if intervention:
        pbm.adjust_rates(topic_weight=1.0, juice_factor=0.5, intervention=True)
    sus, bel, imm = pbm.simulate_step()
    pbm_counts.append(bel)

# Print results
print('Round,ABM_believers,PBM_believers')
for i in range(len(abm_counts)):
    print(f"{i},{abm_counts[i]},{pbm_counts[i]}")

# Also print brief stats
print('\nABM peak:', max(abm_counts), 'at round', abm_counts.index(max(abm_counts)))
print('PBM peak:', max(pbm_counts), 'at round', pbm_counts.index(max(pbm_counts)))
