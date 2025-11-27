def match_class(text: str):
    if not text: 
        return None
    text = text.lower()
    # Tiny mapper (modify for your dataset)
    synonyms = {
        "car": ["car", "vehicle"],
        "plane": ["airplane", "aircraft", "plane"],
        "ship": ["ship", "boat", "vessel"],
        "building": ["building", "house", "structure"]
    }
    for key, words in synonyms.items():
        if any(w in text for w in words):
            return key
    return None