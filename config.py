"""Configuration and constants for the Fake News Simulator."""

# Topic weights (W_topic)
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
    "Phishing": ["Phishing Campaign", "Fake Scholarship Scam", "Emergency VPN Update", "Email Server Compromise"],
    "Reputation-based Rumors": ["Lecturer Scandal", "WiFi Surveillance Rumor", "Fake Exam Timetable", "AI Surveillance Ethics"],
    "Policy Manipulation": ["Student Aid Sabotage", "Ransomware Alert", "Zero-day Exploit", "Data Breach"],
    "Scare Tactics": ["Financial Scam", "University Database Hacked", "Campus Virus Leak"]
}

CATEGORY_TRAIT = {
    "Phishing": "trust_level",
    "Reputation-based Rumors": "confirmation_bias",
    "Policy Manipulation": "critical_thinking",
    "Scare Tactics": "emotional_susceptibility"
}

# --- SIMULATION CONSTANTS (Matching Documentation) ---

# 1. Global Settings
THETA_B = 0.5       # Base Belief Threshold
THETA_S = 0.7       # Base Sharing Threshold
BASE_GAMMA = 0.3    # Base Fact-Check Effectiveness
LAMBDA_DECAY = 0.1  # Base Belief Decay Rate

# 2. Agent Trait Weights (Beta coefficients)
BETA1 = 0.4  # Confirmation Bias Weight
BETA2 = 0.3  # Emotional Susceptibility Weight
BETA3 = 0.2  # Trust Level Weight
BETA4 = 0.3  # Critical Thinking Weight (Resistance)

# 3. Belief Equation Constants (Equation 2.1)
P_BASE_FLOOR = 0.2      # Minimum probability floor
P_BASE_TOPIC = 0.15     # Weight for Topic relevance
P_BASE_JUICE = 0.10     # Weight for Juiciness
T_INFLUENCE_JUICE = 0.2 # Extra juice weight in trait influence
P_STAR_CAP = 0.95       # Maximum pre-social belief probability

# 4. Social & Intervention Constants
FC_GAMMA_MULT = 2.5     # Gamma multiplier during intervention
FC_IMPACT_MULT = 1.5    # Fact-check impact multiplier during intervention
FC_BASE_MULT = 0.8      # Fact-check impact multiplier (normal)
SHARE_INT_PENALTY = 0.45 # Sharing probability multiplier during intervention (0.45 = 55% reduction)

# 5. Stochastic Factors
BOREDOM_PROB = 0.05     # Probability of spontaneous recovery (5%)

# 6. PBM Constants
MAX_NEW_BELIEVER_RATIO = 0.3 # Max % of susceptible that can convert per round