"""Population-based modeling for fake news spread simulation."""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from config import MAX_NEW_BELIEVER_RATIO  # <-- IMPORTED FROM CONFIG

@dataclass
class SimulationRates:
    """Container for simulation rate parameters."""
    contact_rate: float = 0.4    # Rate of contact between individuals
    belief_rate: float = 0.3     # Rate of belief adoption
    recovery_rate: float = 0.1   # Rate of becoming immune

class PopulationSimulator:
    def __init__(self, initial_population: int, initial_believers: int = 2):
        """Initialize the population-based simulator.

        Args:
            initial_population: total population size
            initial_believers: initial number of believers in the population
        """
        self.total_population = initial_population
        self.rates = SimulationRates()

        # Initialize populations with configurable initial believers
        self.initial_believers = max(0, int(initial_believers))
        self.susceptible = initial_population - self.initial_believers
        self.believers = int(self.initial_believers)
        self.immune = 0
        
        # Track history
        self.history = {
            'susceptible': [self.susceptible],
            'believers': [self.believers],
            'immune': [self.immune]
        }
        
    def adjust_rates(self, topic_weight: float, juice_factor: float, 
                    intervention: bool = False) -> None:
        """
        Adjusts rates based on conditions.
        This will respect user slider settings for contact/belief,
        but will always apply intervention logic to the recovery rate.
        """
        
        # Check if contact_rate is still at its default (0.4)
        # If not, the user's slider value is respected.
        if round(self.rates.contact_rate, 2) == 0.40:
            base_contact = 0.4
            topic_effect = max(0.1, min(0.8, topic_weight))
            self.rates.contact_rate = min(0.8, base_contact * (1 + 0.3 * topic_effect))

        # Check if belief_rate is still at its default (0.3)
        # If not, the user's slider value is respected.
        if round(self.rates.belief_rate, 2) == 0.30:
            base_belief = 0.3
            juice_effect = max(0.1, min(0.7, juice_factor))
            self.rates.belief_rate = min(0.7, base_belief * (1 + 0.4 * juice_effect))
        
        # --- This is the key logic ---
        # This function is called at the start (intervention=False)
        # and again when the intervention hits (intervention=True).
        
        if intervention:
            # When the intervention hits, ALWAYS apply a recovery rate.
            # This is what you find logical, and it is.
            self.rates.recovery_rate = 0.1 * 1.5 # Apply 0.15 recovery
        else:
            # This runs at the start. If the rate is still the default 0.1,
            # it sets it (before the sliders override it).
            if round(self.rates.recovery_rate, 2) == 0.10:
                 self.rates.recovery_rate = 0.1 * 1.0
        
    def simulate_step(self) -> Tuple[int, int, int]:
        """Run one step of the simulation using SIR model equations."""
        # Current state
        N = self.total_population
        S, I, R = self.susceptible, self.believers, self.immune
        
        # Calculate state changes using SIR model with dampening
        exposure_rate = (self.rates.contact_rate * self.rates.belief_rate * S * I) / N
        new_believers = min(exposure_rate, S * MAX_NEW_BELIEVER_RATIO)  # Cap maximum new believers
        recoveries = self.rates.recovery_rate * I
        
        # Update state with minimum bounds
        self.susceptible = max(0, S - new_believers)
        self.believers = max(0, I + new_believers - recoveries)
        self.immune = R + recoveries
        
        # Update history
        self.history['susceptible'].append(int(self.susceptible))
        self.history['believers'].append(int(self.believers))
        self.history['immune'].append(int(self.immune))
        
        return (int(self.susceptible), int(self.believers), int(self.immune))
    
    def get_history(self) -> Dict[str, List[int]]:
        """Get the simulation history."""
        return self.history
