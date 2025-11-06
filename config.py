"""Configuration and constants for the Fake News Simulator."""

# Topic weights
TOPICS = {
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

# Topic categories and their main agent trait influence
TOPIC_CATEGORIES = {
    "Phishing": [
        "Phishing Campaign", "Fake Scholarship Scam", "Emergency VPN Update", "Email Server Compromise"
    ],
    "Reputation-based Rumors": [
        "Lecturer Scandal", "WiFi Surveillance Rumor", "Fake Exam Timetable", "AI Surveillance Ethics"
    ],
    "Policy Manipulation": [
        "Student Aid Sabotage", "Ransomware Alert", "Zero-day Exploit", "Data Breach"
    ],
    "Scare Tactics": [
        "Financial Scam", "University Database Hacked", "Campus Virus Leak"
    ]
}

# Map category to main trait
CATEGORY_TRAIT = {
    "Phishing": "trust_level",
    "Reputation-based Rumors": "confirmation_bias",
    "Policy Manipulation": "critical_thinking",
    "Scare Tactics": "emotional_susceptibility"
}

# Simulation parameters
THETA_B = 0.5  # Belief threshold
THETA_S = 0.7  # Sharing threshold
A = 0.6        # Exposure curve steepness
BASE_GAMMA = 0.3  # Base fact-checking effectiveness
LAMBDA_DECAY = 0.1  # Belief decay rate
DELTA_T = 1.0  # Time step
BETA1, BETA2, BETA3, BETA4 = 0.4, 0.3, 0.2, 0.3  # Trait weights