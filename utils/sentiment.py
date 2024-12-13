class AnimeSentimentAnalyzer:
    """Analyzes sentiment in user messages about anime"""
    
    def __init__(self):
        self.positive_words = {
            "bagus", "suka", "keren", "mantap", "hebat", "wow", "amazing",
            "recommended", "masterpiece", "favorite", "terbaik", "sempurna",
            "enjoy", "seru", "menarik", "recommendable", "worth", "layak"
        }
        
        self.negative_words = {
            "jelek", "buruk", "bosan", "membosankan", "tidak suka", "skip",
            "waste", "buang waktu", "disappointed", "kecewa", "overrated",
            "tidak bagus", "tidak worth", "tidak layak", "dropped"
        }
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text, returns score between -1 and 1
        -1 = very negative, 1 = very positive
        """
        words = text.lower().split()
        score = 0
        word_count = 0
        
        for word in words:
            if word in self.positive_words:
                score += 1
                word_count += 1
            elif word in self.negative_words:
                score -= 1
                word_count += 1
        
        return score / (word_count if word_count > 0 else 1)