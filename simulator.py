"""Simulation logic for the fake news spread model.

This is the canonical `simulator.py` with probabilistic sharing and transmission attribution.
"""

import numpy as np
import networkx as nx
import random
import pandas as pd
from config import (THETA_B, THETA_S, BASE_GAMMA, LAMBDA_DECAY,
                    BETA1, BETA2, BETA3, BETA4, CATEGORY_TRAIT,
                    P_BASE_FLOOR, P_BASE_TOPIC, P_BASE_JUICE, T_INFLUENCE_JUICE,
                    P_STAR_CAP, FC_GAMMA_MULT, FC_IMPACT_MULT, FC_BASE_MULT,
                    SHARE_INT_PENALTY, BOREDOM_PROB)


class FakeNewsSimulator:
    def __init__(self, num_agents, agent_data=None):
        """Initialize the simulator with given number of agents and optional agent data."""
        self.num_agents = num_agents
        # Network density of 0.2 - average person knows 20% of the network
        self.G = nx.erdos_renyi_graph(num_agents, 0.2)
        self.agent_states = {}
        self._graph_pos = nx.spring_layout(self.G, seed=42)

        if agent_data is None:
            agent_data = self._generate_random_agents(num_agents)
        self._initialize_agents(agent_data)

    def _generate_random_agents(self, n):
        """Generate random agent profiles."""
        data = {
            'confirmation_bias': np.random.beta(5, 5, n),  # Bell curve around 0.5
            'emotional_susceptibility': np.random.beta(5, 5, n),
            'trust_level': np.random.beta(4, 6, n),  # Slightly skeptical
            'critical_thinking': np.random.beta(6, 4, n),  # Slightly higher
            'fact_check_signal': np.random.beta(4, 6, n),  # Most don't fact check much
            'risk_perception': np.random.beta(5, 5, n)
        }
        return pd.DataFrame(data)

    def _initialize_agents(self, agent_df):
        """Initialize agent states from provided data."""
        for i, row in agent_df.iterrows():
            risk_perception = row['risk_perception'] if 'risk_perception' in row else 0.5
            self.agent_states[i] = {
                'belief': 0.0,
                'shared': False,
                'confirmation_bias': row.get('confirmation_bias', 0.5),
                'emotional_susceptibility': row.get('emotional_susceptibility', 0.5),
                'trust_level': row.get('trust_level', 0.5),
                'critical_thinking': row.get('critical_thinking', 0.5),
                'fact_check_signal': row.get('fact_check_signal', 0.5),
                'risk_perception': risk_perception,
                'scammed': False,
                # Immune means the agent will no longer be susceptible to this fake news
                'immune': False
            }

    def seed_initial_state(self, is_scam=False):
        """Set initial believers/scam victims."""
        for node in random.sample(list(self.agent_states.keys()), min(2, len(self.agent_states))):
            if is_scam:
                self.agent_states[node]['scammed'] = True
            else:
                self.agent_states[node]['belief'] = 1.0
                self.agent_states[node]['shared'] = True

    def simulate_fake_news_round(self, juice_factor, topic_weight, topic_category,
                                 intervention=False, extra_rounds=False):
        """Simulate one round of fake news spread."""
        nodes = list(self.G.nodes())
        
        # USE CONSTANT: Fact-check gamma multiplier (FC_GAMMA_MULT)
        gamma = BASE_GAMMA * (FC_GAMMA_MULT if intervention else 1.0)
        
        shared = {i: self.agent_states[i]['shared'] for i in nodes}
        beliefs = {i: self.agent_states[i]['belief'] for i in nodes}
        new_beliefs = {}

        # Get thresholds and parameters based on conditions
        theta_b, theta_s, lambda_decay, exposure_alpha = self._get_dynamic_parameters(
            juice_factor, intervention, extra_rounds
        )

        # Calculate weights based on topic category
        weights = self._calculate_trait_weights(topic_category)

        # Update beliefs for each agent
        for i in nodes:
            neighbors = list(self.G.neighbors(i))
            exposures = sum(shared.get(j, False) for j in neighbors)
            a = self.agent_states[i]

            # 1. Calculate P_star (Pre-social probability)
            P_star = self._calculate_belief_probability(
                a, weights, topic_weight, juice_factor
            )
            
            # 2. Calculate Social Factor
            social_factor = (1 / (1 + np.exp(-exposure_alpha * exposures)))
            exposure_factor = min(0.8, social_factor * (1 + 0.35 * juice_factor))
            
            # 3. Calculate Fact-Check Impact
            # USE CONSTANTS: FC_IMPACT_MULT and FC_BASE_MULT
            fact_check_impact = gamma * a.get('fact_check_signal', 0.0) * (FC_IMPACT_MULT if intervention else FC_BASE_MULT)
            
            # 4. Final Belief Probability
            P_believe = (exposure_factor * P_star) - fact_check_impact
            P_believe = max(0.0, P_believe)

            # --- Probabilistic Recovery (Boredom) ---
            # USE CONSTANT: BOREDOM_PROB
            if beliefs.get(i, 0.0) > theta_b and random.random() < BOREDOM_PROB:
                new_beliefs[i] = 0.0
                # Optional: Make them immune so they don't instantly rejoin
                # self.agent_states[i]['immune'] = True 
            
            # If no exposure, force decay
            elif extra_rounds or exposures == 0:
                decay_modifier = max(0.6, 1.0 - (exposures / 8))
                new_beliefs[i] = beliefs.get(i, 0.0) * np.exp(-lambda_decay * decay_modifier)
            
            # Standard Update Logic
            elif P_believe > theta_b:
                # Remove the +0.1 bonus to stop artificial boosting
                new_beliefs[i] = min(0.95, P_believe) 
            else:
                decay_modifier = max(0.5, 1.0 - (exposures / 10))
                new_beliefs[i] = beliefs.get(i, 0.0) * np.exp(-lambda_decay * decay_modifier)

        # Prepare transmission logging for this round
        transmissions = []  # list of (source, target)

        # Determine new belief and probabilistic sharing for each agent
        prev_belief_flags = {i: (beliefs.get(i, 0.0) > theta_b) for i in nodes}

        # Update agent beliefs
        INERTIA = getattr(self, 'belief_inertia', 0.6)  # how much old belief persists (0..1)
        MAX_DELTA = getattr(self, 'max_belief_delta', 0.2)  # max allowed change per round

        # Immunity check
        for i in nodes:
            if prev_belief_flags.get(i, False) and new_beliefs.get(i, 0.0) <= theta_b:
                self.agent_states[i]['immune'] = True
                new_beliefs[i] = 0.0

        for i in nodes:
            if self.agent_states[i].get('immune', False):
                self.agent_states[i]['belief'] = 0.0
                self.agent_states[i]['shared'] = False
                continue

            old_b = beliefs.get(i, 0.0)
            target_b = new_beliefs.get(i, 0.0)
            delta = target_b - old_b
            if delta > MAX_DELTA: delta = MAX_DELTA
            elif delta < -MAX_DELTA: delta = -MAX_DELTA
            
            smoothed = old_b + (1.0 - INERTIA) * delta
            self.agent_states[i]['belief'] = float(max(0.0, min(1.0, smoothed)))

        # Compute probabilistic sharing
        def _sig(x):
            return 1.0 / (1.0 + np.exp(-x))

        ALPHA = getattr(self, 'share_belief_weight', 3.0)
        BETA_EMO = getattr(self, 'share_emotional_weight', 1.2)
        BETA_CONF = getattr(self, 'share_confirmation_weight', 0.8)
        DELTA_JUICE = getattr(self, 'share_juice_weight', 0.8)
        ETA_CRIT = getattr(self, 'share_critical_penalty', 2.5)
        OFFSET = getattr(self, 'share_offset', -1.0)

        new_shared = {}
        for i in nodes:
            a = self.agent_states[i]
            bval = self.agent_states[i].get('belief', 0.0)
            score = (ALPHA * bval
                     + BETA_EMO * a.get('emotional_susceptibility', 0.0)
                     + BETA_CONF * a.get('confirmation_bias', 0.0)
                     + DELTA_JUICE * juice_factor
                     - ETA_CRIT * a.get('critical_thinking', 0.0)
                     + OFFSET)
            p_share = float(_sig(score))
            p_share = p_share * max(0.0, bval)
            
            # USE CONSTANT: SHARE_INT_PENALTY
            if intervention:
                p_share *= SHARE_INT_PENALTY
            
            shared_decision = (random.random() < p_share)
            new_shared[i] = shared_decision

        # Attribute new believers
        for i in nodes:
            neigh = list(self.G.neighbors(i))
            became_believer = (not prev_belief_flags.get(i, False)) and (new_beliefs.get(i, 0.0) > theta_b)
            if became_believer:
                source_candidates = [j for j in neigh if shared.get(j, False)]
                if source_candidates:
                    mode = getattr(self, 'attribution_mode', 'weighted')
                    if mode == 'random':
                        src = random.choice(source_candidates)
                    elif mode == 'first':
                        src = source_candidates[0]
                    else:  # weighted attribution
                        weights_src = []
                        for j in source_candidates:
                            w = max(0.01, beliefs.get(j, 0.0))
                            weights_src.append(w)
                        total_w = sum(weights_src)
                        if total_w <= 0:
                            src = random.choice(source_candidates)
                        else:
                            probs = [w / total_w for w in weights_src]
                            src = np.random.choice(source_candidates, p=probs)
                    transmissions.append((int(src), int(i)))

        for i, val in new_shared.items():
            self.agent_states[i]['shared'] = bool(val)

        if not hasattr(self, 'transmission_history'):
            self.transmission_history = []
        self.transmission_history.append(transmissions)
        self.last_transmissions = transmissions

        return [self.agent_states[i]['shared'] for i in nodes]

    def _get_dynamic_parameters(self, juice_factor, intervention, extra_rounds):
        """Get dynamic parameters based on current conditions."""
        # Realistic thresholds based on news characteristics
        if juice_factor >= 0.95:  # Very juicy/viral news
            theta_b = 0.35  # Easier to believe
            theta_s = 0.55  # Easier to share
            lambda_decay = 0.05  # Slower decay
            exposure_alpha = 1.0  # Strong social influence
        elif juice_factor >= 0.8:  # Moderately viral
            theta_b = 0.42
            theta_s = 0.62
            lambda_decay = 0.08
            exposure_alpha = 0.8
        else:  # Regular news
            theta_b = 0.48  # Moderate threshold
            theta_s = 0.65
            lambda_decay = 0.1
            exposure_alpha = 0.7

        # Intervention effects
        if intervention:
            theta_s *= 1.3  # 30% harder to share
            lambda_decay *= 1.5  # 50% faster decay

        # Natural decay in extra rounds
        if extra_rounds:
            lambda_decay *= 1.2  # 20% faster decay

        return theta_b, theta_s, lambda_decay, exposure_alpha

    def _calculate_trait_weights(self, topic_category):
        """Calculate trait weights based on topic category."""
        weights = {
            'confirmation_bias': BETA1,
            'emotional_susceptibility': BETA2,
            'trust_level': BETA3,
            'critical_thinking': BETA4
        }

        if topic_category and topic_category in CATEGORY_TRAIT:
            main_trait = CATEGORY_TRAIT[topic_category]
            for k in weights:
                # More balanced trait influence
                weights[k] *= 1.4 if k == main_trait else 0.9

        return weights

    def _calculate_belief_probability(self, agent, weights, topic_weight, juice_factor):
        """Calculate the probability of an agent believing the fake news."""
        
        # USE CONSTANT: T_INFLUENCE_JUICE
        trait_influence = (
            weights['confirmation_bias'] * agent.get('confirmation_bias', 0.5) +
            weights['emotional_susceptibility'] * agent.get('emotional_susceptibility', 0.5) * (topic_weight + T_INFLUENCE_JUICE * juice_factor) +
            weights['trust_level'] * agent.get('trust_level', 0.5)
        )

        resistance = weights['critical_thinking'] * agent.get('critical_thinking', 0.5)

        # USE CONSTANTS: P_BASE_FLOOR, P_BASE_TOPIC, P_BASE_JUICE
        base_prob = P_BASE_FLOOR + (P_BASE_TOPIC * topic_weight) + (P_BASE_JUICE * juice_factor)

        # USE CONSTANT: P_STAR_CAP
        return min(P_STAR_CAP, max(0.0, base_prob + trait_influence - resistance))

    def simulate_scam_round(self):
        """Simulate one round of scam spread."""
        nodes = list(self.G.nodes())
        scammed = {i: self.agent_states[i]['scammed'] for i in nodes}
        new_scammed = {}

        for i in nodes:
            if scammed[i]:
                new_scammed[i] = True
                continue

            neighbors = list(self.G.neighbors(i))
            exposures = sum(scammed[j] for j in neighbors)

            if exposures == 0:
                new_scammed[i] = False
            else:
                a = self.agent_states[i]
                P_scam = 0.5 * a.get('trust_level', 0.5) + 0.4 * (1 - a.get('risk_perception', 0.5)) - 0.3 * a.get('critical_thinking', 0.5)
                P_scam = max(0, min(1, P_scam))
                new_scammed[i] = P_scam > 0.5

        for i in nodes:
            self.agent_states[i]['scammed'] = new_scammed[i]

        return [self.agent_states[i]['scammed'] for i in nodes]

    def get_graph_layout(self):
        """Get the precomputed graph layout."""
        return self._graph_pos

    def get_node_colors(self, is_scam=False):
        """Get colors for visualization."""
        if is_scam:
            return ['red' if self.agent_states[n]['scammed'] else 'blue'
                    for n in self.G.nodes()]
        return ['red' if self.agent_states[n]['shared'] else 'blue'
                for n in self.G.nodes()]