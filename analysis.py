"""Utility functions for analyzing fake news context and topics."""

def analyze_context_juiciness(context):
    """Analyze the juiciness of the fake news context."""
    very_high = [
        "virus", "covid", "authorities silent", "for sale", "emergency", "block access",
        "identity theft", "sabotage", "privacy violated"
    ]
    high = [
        "hacked", "hack", "leaked", "leak", "scandal", "urgent", "phishing", "compromised",
        "password reset", "plagiarism", "fake exam", "panic", "malware", "fraud", "scam",
        "stolen", "for sale", "dark web", "delayed", "failure to comply"
    ]
    med_high = [
        "vpn update", "proxy", "dissatisfaction", "distrust", "sabotage", "privacy",
        "protest", "mass sharing", "grant", "scholarship", "register now", "financial aid"
    ]
    medium = [
        "warning", "breach", "incident", "risk", "threat", "phishing", "data", "security",
        "fake", "rumor", "monitored", "encrypted", "browser", "campus wifi",
        "ai surveillance", "ethical", "concerns"
    ]
    low = [
        "update", "news", "report", "statement", "info", "information", "timetable", "verify"
    ]
    
    context = context.lower()
    score = 20  # default
    
    if any(word in context for word in very_high):
        return 98
    if any(word in context for word in high):
        return 90
    if any(word in context for word in med_high):
        return 75
    if any(word in context for word in medium):
        return 60
    if any(word in context for word in low):
        return 30
    return score

def infer_topic_from_context(context, topics_dict, topic_categories):
    """Infer the topic and its properties from the context."""
    topic_keywords = {
        "Ransomware Alert": ["ransomware", "ransom"],
        "Data Breach": ["breach", "leak", "data"],
        "Zero-day Exploit": ["zero-day", "exploit", "vulnerability"],
        "Phishing Campaign": ["phishing", "phish", "email", "campaign"],
        "Financial Scam": ["scam", "fraud", "money", "bank", "finance", "financial"],
        "University Database Hacked": ["university database hacked", "student records", "dark web"],
        "Fake Scholarship Scam": ["scholarship", "register now", "uum id", "grant"],
        "Emergency VPN Update": ["vpn update", "block access"],
        "Email Server Compromise": ["email server", "compromised", "password reset"],
        "Lecturer Scandal": ["lecturer", "plagiarism", "scandal", "leaked documents"],
        "WiFi Surveillance Rumor": ["wifi", "monitored", "proxy", "encrypted browsers"],
        "Fake Exam Timetable": ["fake exam", "timetable", "verify"],
        "Student Aid Sabotage": ["financial aid", "delayed", "sabotage"],
        "AI Surveillance Ethics": ["ai surveillance", "privacy rights", "violated"],
        "Campus Virus Leak": ["virus", "covid", "authorities silent"]
    }
    
    context = context.lower()
    for topic, keywords in topic_keywords.items():
        if any(word in context for word in keywords):
            # Find category
            for cat, topic_list in topic_categories.items():
                if topic in topic_list:
                    return topic, topics_dict[topic], cat
            return topic, topics_dict[topic], None
    
    return "Other", 1.0, None  # Default weight and no category
