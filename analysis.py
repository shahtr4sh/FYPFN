"""
Text analysis module for the Fake News Simulator.
analyzes context strings to determine topic weights and juiciness scores.
"""

import re
# Import the new configuration dictionaries
from config import TOPICS, TOPIC_CATEGORIES, JUICINESS_KEYWORDS

def analyze_context_juiciness(context_text):
    """
    Calculates a 'Juiciness' score (0-100) based on the presence of viral keywords.
    
    Args:
        context_text (str): The news headline or context string entered by the user.
        
    Returns:
        int: A score between 0 and 100.
    """
    if not context_text:
        return 50  # Default neutral score if empty

    text_upper = context_text.upper()
    score = 0
    
    # 1. Base Score calculation using the Dictionary from config.py
    # We iterate through every keyword in your new list
    for keyword, value in JUICINESS_KEYWORDS.items():
        if keyword in text_upper:
            score += value
            
    # 2. Add bonus for exclamation marks (max +15)
    exclamation_count = text_upper.count('!')
    score += min(15, exclamation_count * 5)
    
    # 3. Add bonus for ALL CAPS words (simple heuristic)
    # Splits text and counts words that are fully uppercase and longer than 3 letters
    words = text_upper.split()
    all_caps_count = sum(1 for w in words if w.isupper() and len(w) > 3)
    score += min(20, all_caps_count * 5)

    # 4. Clamp the result between 0 and 100
    # (We ensure it doesn't go below 10 or above 100)
    final_score = max(10, min(100, score))
    
    return final_score

def infer_topic_from_context(context_text, topics_dict=TOPICS, categories_dict=TOPIC_CATEGORIES):
    """
    Matches the input text against known topics to find the best fit.
    
    Args:
        context_text (str): The user input.
        topics_dict (dict): The dictionary of topics and their base weights.
        categories_dict (dict): The dictionary mapping categories to topics.
        
    Returns:
        tuple: (Best Topic Name, Topic Weight, Topic Category)
    """
    if not context_text:
        return "General Rumor", 1.0, "Uncategorized"

    text_lower = context_text.lower()
    best_topic = "General Rumor"
    best_weight = 1.0
    
    # Simple keyword matching: find the longest matching topic name in the text
    # (e.g., if text contains "Ransomware Alert", we match that topic)
    found = False
    for topic, weight in topics_dict.items():
        # Check if the topic string itself is in the text (e.g. "Data Breach")
        # OR check if the text contains keywords from the topic name
        topic_keywords = topic.lower().split()
        
        # If the exact topic phrase is in the text
        if topic.lower() in text_lower:
            best_topic = topic
            best_weight = weight
            found = True
            break
            
        # Fallback: if 2+ words from the topic name appear (e.g. "University" and "Hacked")
        match_count = sum(1 for word in topic_keywords if word in text_lower)
        if match_count >= 2:
            best_topic = topic
            best_weight = weight
            found = True
            # We don't break yet, in case a better match comes later, 
            # but usually first match is fine for this scope.
    
    # Determine Category
    category = "Uncategorized"
    for cat, topic_list in categories_dict.items():
        if best_topic in topic_list:
            category = cat
            break
            
    return best_topic, best_weight, category