"""Configuration and constants for the Fake News Simulator."""

# Topic weights (W_topic)
"""Configuration and constants for the Fake News Simulator."""

# --- ENHANCED TOPIC WEIGHTS (W_topic) ---
# Higher weight (e.g., 1.5) = Inherently more believable/viral
TOPICS = {
    # 1. Cybersecurity & Tech (High technical fear)
    "Ransomware Alert": 1.0,
    "Data Breach": 1.1,
    "Zero-day Exploit": 1.2,
    "Emergency VPN Update": 1.1,
    "Email Server Compromise": 1.2,
    "AI Surveillance Ethics": 1.1,
    "5G Radiation Leak": 1.4,
    "Deepfake Voice Scam": 1.3,

    # 2. Campus & University Life (High local relevance)
    "University Database Hacked": 1.3,
    "Fake Scholarship Scam": 1.4,
    "Lecturer Scandal": 1.3,
    "WiFi Surveillance Rumor": 1.1,
    "Fake Exam Timetable": 1.4,
    "Student Aid Sabotage": 1.2,
    "Tuition Fee Hike Leak": 1.5,
    "Hostel Eviction Notice": 1.4,
    "Campus Bus Strike": 1.2,

    # 3. Health & Safety (High emotional fear)
    "Campus Virus Leak": 1.5,
    "Cafeteria Food Poisoning": 1.4,
    "New Pandemic Variant": 1.3,
    "Toxic Water Supply": 1.4,
    "Free Vaccine Side Effects": 1.2,
    "Mental Health Crisis Coverup": 1.3,

    # 4. Financial & Scams (High greed/desperation)
    "Financial Scam": 1.0,
    "PTPTN Loan Waiver": 1.5,
    "Crypto Investment Scheme": 1.1,
    "E-Wallet Hack Warning": 1.3,
    "Free Laptop Grant": 1.4,

    # 5. Social & Political (High polarization)
    "Student Council Rigging": 1.3,
    "Protest Organization": 1.2,
    "Dress Code Crackdown": 1.3,
    "Religious Society Ban": 1.4
}

# --- ENHANCED CATEGORIES ---
TOPIC_CATEGORIES = {
    "Cybersecurity": [
        "Ransomware Alert", "Data Breach", "Zero-day Exploit", 
        "Emergency VPN Update", "Email Server Compromise", "Deepfake Voice Scam"
    ],
    "Campus Rumors": [
        "Lecturer Scandal", "WiFi Surveillance Rumor", "Fake Exam Timetable", 
        "Tuition Fee Hike Leak", "Hostel Eviction Notice", "Campus Bus Strike",
        "Student Council Rigging", "Dress Code Crackdown"
    ],
    "Financial Scams": [
        "Financial Scam", "Fake Scholarship Scam", "Student Aid Sabotage", 
        "PTPTN Loan Waiver", "Crypto Investment Scheme", "Free Laptop Grant"
    ],
    "Health Scare": [
        "Campus Virus Leak", "Cafeteria Food Poisoning", "New Pandemic Variant", 
        "Toxic Water Supply", "5G Radiation Leak", "Free Vaccine Side Effects"
    ]
}

# --- PRIMARY TRAIT MAPPING ---
# Maps the category to the agent trait that makes them most vulnerable
CATEGORY_TRAIT = {
    "Cybersecurity": "trust_level",              # Relies on trusting the "System Admin"
    "Campus Rumors": "confirmation_bias",        # Relies on pre-existing biases against authority
    "Financial Scams": "critical_thinking",      # Relies on lack of scrutiny (Greed overrides logic)
    "Health Scare": "emotional_susceptibility"   # Relies on pure fear/panic
}

# --- BALANCED JUICINESS KEYWORDS (config.py) ---
JUICINESS_KEYWORDS = {
    # --- TIER 1: NUCLEAR / LIFE THREATENING (30 Points Each) ---
    # Hit 3 of these and you're basically at 90-100%.
    "EXPLOSION": 30, "BOMB": 30, "TERRORIST": 30, "SHOOTING": 30,
    "DEADLY": 30, "DEATH": 30, "KILLED": 30, "SUICIDE": 30,
    "COLLAPSE": 30, "VIRUS": 30, "PANDEMIC": 30, "OUTBREAK": 30,
    "RAPE": 30, "MURDER": 30, "KIDNAP": 30, "HOSTAGE": 30, "HOSPITALIZED": 30,

    # --- TIER 2: HIGH URGENCY / CRIME (15 Points Each) ---
    "BREAKING": 15, "URGENT": 15, "WARNING": 15, "ALERT": 15,
    "EMERGENCY": 15, "LOCKDOWN": 15, "EVACUATE": 15, "ARRESTED": 15,
    "STOLEN": 15, "HACKED": 15, "SCAM": 15, "LEAKED": 15,
    "BANKRUPT": 15, "RIGGED": 15, "CORRUPTION": 15, "SCANDAL": 15, "COVERUP": 15,

    # --- TIER 3: ACADEMIC & ADMIN TRIGGERS (10 Points Each) ---
    "BANNED": 10, "SUSPENDED": 10, "EXPELLED": 10, "FIRED": 10,
    "CANCELLED": 10, "INCREASE": 10, "HIKE": 10, "CUT": 10,
    "OFFICIAL": 10, "POLICE": 10, "MINISTRY": 10, "VC": 10,
    "FREE": 10, "CASH": 10, "MONEY": 10, "GRANT": 10, "CLAIM": 10, "LINK": 10,

    # --- TIER 4: CONTEXT FILLERS (5 Points Each) ---
    "STUDENT": 5, "CAMPUS": 5, "HOSTEL": 5, "EXAM": 5, 
    "RESULT": 5, "WIFI": 5, "SYSTEM": 5, "BUS": 5,
    "MADANI": 5, "PTPTN": 5
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