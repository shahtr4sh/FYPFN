"""Base class for simulation models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

class BaseSimulationModel(ABC):
    def __init__(self, population_size: int):
        """Initialize base simulation model.
        
        Args:
            population_size: Total number of agents/individuals
        """
        self.population_size = population_size
        self.current_round = 0
        self.history: Dict[str, List[Any]] = {}
        
    @abstractmethod
    def simulate_step(self) -> Dict[str, Any]:
        """Run one step of simulation."""
        pass
        
    @abstractmethod
    def reset(self) -> None:
        """Reset simulation state."""
        pass
        
    @abstractmethod
    def adjust_parameters(self, **kwargs) -> None:
        """Adjust simulation parameters."""
        pass
        
    def get_history(self) -> Dict[str, List[Any]]:
        """Get simulation history."""
        return self.history
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state."""
        return {key: values[-1] if values else None 
                for key, values in self.history.items()}