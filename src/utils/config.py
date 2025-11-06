"""Configuration and constants for the Fake News Simulator."""

from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SimulationParameters:
    """Core simulation parameters."""
    THETA_B: float = 0.5  # Belief threshold
    THETA_S: float = 0.7  # Sharing threshold
    A: float = 0.6        # Exposure curve steepness
    BASE_GAMMA: float = 0.3  # Base fact-checking effectiveness
    LAMBDA_DECAY: float = 0.1  # Belief decay rate
    DELTA_T: float = 1.0  # Time step
    
    # Trait weights
    BETA1: float = 0.4  # Confirmation bias weight
    BETA2: float = 0.3  # Emotional susceptibility weight
    BETA3: float = 0.2  # Trust level weight
    BETA4: float = 0.3  # Critical thinking weight

class TopicConfiguration:
    """Topic-related configuration."""
    
    # Topic weights
    TOPICS: Dict[str, float] = {
        "Ransomware Alert": 1.0,
        "Data Breach": 0.9,
        "Zero-day Exploit": 1.2,
        "Phishing Campaign": 1.1,
        "Financial Scam": 1.0,
        "University Database Hacked": 1.3,
        "Fake Scholarship Scam": 1.4,
        "Emergency VPN Update": 1.2,
        "Email Server Compromise": 1.2,
        "Lecturer Scandal": 1.3,
        "WiFi Surveillance Rumor": 1.0,
        "Fake Exam Timetable": 1.4,
        "Student Aid Sabotage": 1.1,
        "AI Surveillance Ethics": 1.0,
        "Campus Virus Leak": 1.5
    }
    
    # Topic categories
    TOPIC_CATEGORIES: Dict[str, List[str]] = {
        "Phishing": [
            "Phishing Campaign", "Fake Scholarship Scam",
            "Emergency VPN Update", "Email Server Compromise"
        ],
        "Reputation-based Rumors": [
            "Lecturer Scandal", "WiFi Surveillance Rumor",
            "Fake Exam Timetable", "AI Surveillance Ethics"
        ],
        "Policy Manipulation": [
            "Student Aid Sabotage", "Ransomware Alert",
            "Zero-day Exploit", "Data Breach"
        ],
        "Scare Tactics": [
            "Financial Scam", "University Database Hacked",
            "Campus Virus Leak"
        ]
    }
    
    # Category to trait mapping
    CATEGORY_TRAIT: Dict[str, str] = {
        "Phishing": "trust_level",
        "Reputation-based Rumors": "confirmation_bias",
        "Policy Manipulation": "critical_thinking",
        "Scare Tactics": "emotional_susceptibility"
    }

class GuiConfiguration:
    """GUI-related configuration."""
    
    # Window dimensions
    MAIN_WINDOW_SIZE = "1200x800"
    SUMMARY_WINDOW_SIZE = "600x800"
    COMPARISON_WINDOW_SIZE = "1200x800"
    
    # Plot configuration
    PLOT_DPI = 100
    PLOT_FIGSIZE = (12, 6)
    
    # Colors
    COLORS = {
        "believers": "red",
        "non_believers": "blue",
        "immune": "green",
        "grid": "#E0E0E0",
        "background": "#F5F5F5"
    }
    
    # Font settings
    FONTS = {
        "title": ("Arial", 12, "bold"),
        "subtitle": ("Arial", 10, "bold"),
        "text": ("Arial", 10),
        "small": ("Arial", 8)
    }

# Create default instances
sim_params = SimulationParameters()
topics = TopicConfiguration()
gui_config = GuiConfiguration()