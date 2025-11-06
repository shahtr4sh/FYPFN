"""Population-based modeling for fake news spread simulation."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .base_model import BaseSimulationModel

@dataclass
class ModelParameters:
    """Model parameters configuration."""
    contact_rate: float = 0.3
    belief_rate: float = 0.4
    recovery_rate: float = 0.1
    initial_believers: int = 2

class PopulationSimulator(BaseSimulationModel):
    def __init__(self, population_size: int, params: Optional[ModelParameters] = None):
        """Initialize the population-based simulator.
        
        Args:
            population_size: Total population size
            params: Optional model parameters
        """
        super().__init__(population_size)
        self.params = params or ModelParameters()
        
        # Initialize populations
        self.reset()
        
    def reset(self) -> None:
        """Reset simulation state."""
        self.susceptible = self.population_size - self.params.initial_believers
        self.believers = self.params.initial_believers
        self.immune = 0
        
        # Reset history
        self.history = {
            'susceptible': [self.susceptible],
            'believers': [self.believers],
            'immune': [self.immune]
        }
        
    def adjust_parameters(self, **kwargs) -> None:
        """Adjust model parameters based on news properties."""
        topic_weight = kwargs.get('topic_weight', 1.0)
        juice_factor = kwargs.get('juice_factor', 0.5)
        intervention = kwargs.get('intervention', False)
        
        # Update rates with constraints
        self.params.contact_rate = min(0.95, 0.3 * topic_weight)
        self.params.belief_rate = min(0.95, 0.4 * (1 + juice_factor))
        self.params.recovery_rate = 0.1 * (2 if intervention else 1)
        
    def simulate_step(self) -> Dict[str, int]:
        """Run one step of the simulation using SIR model equations.
        
        Returns:
            Dict containing current population counts
        """
        # Calculate transitions using SIR model
        N = self.population_size
        S, I, R = self.susceptible, self.believers, self.immune
        
        # Calculate state changes
        new_believers = self._calculate_new_believers(S, I, N)
        recoveries = self._calculate_recoveries(I)
        
        # Update states
        self.susceptible = max(0, S - new_believers)
        self.believers = max(0, I + new_believers - recoveries)
        self.immune = R + recoveries
        
        # Update history
        self._update_history()
        
        return self._get_current_counts()
        
    def _calculate_new_believers(self, S: float, I: float, N: float) -> float:
        """Calculate new believers using SIR model equation."""
        new_believers = (self.params.contact_rate * 
                        self.params.belief_rate * S * I / N)
        return min(new_believers, S)  # Cannot exceed susceptible population
        
    def _calculate_recoveries(self, I: float) -> float:
        """Calculate recoveries using recovery rate."""
        return self.params.recovery_rate * I
        
    def _update_history(self) -> None:
        """Update simulation history."""
        self.history['susceptible'].append(int(self.susceptible))
        self.history['believers'].append(int(self.believers))
        self.history['immune'].append(int(self.immune))
        
    def _get_current_counts(self) -> Dict[str, int]:
        """Get current population counts."""
        return {
            'susceptible': int(self.susceptible),
            'believers': int(self.believers),
            'immune': int(self.immune)
        }
        
    def get_statistics(self) -> Dict[str, float]:
        """Calculate key statistics from simulation history.
        
        Returns:
            Dictionary with statistics
        """
        believers = self.history['believers']
        peak_believers = max(believers)
        final_immune = self.history['immune'][-1]
        
        return {
            'peak_believers': peak_believers,
            'final_immune': final_immune,
            'spread_rate': peak_believers / self.population_size,
            'susceptible_remaining': self.susceptible
        }