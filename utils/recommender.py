
from typing import List, Dict, Set
from collections import defaultdict
import numpy as np
from data.knowledge_base import AnimeKnowledgeBase

class AnimeRecommender:
    """Advanced recommendation system for anime"""
    
    def __init__(self, knowledge_base: AnimeKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.genre_weights = defaultdict(float)
        self.update_genre_weights()
    
    def update_genre_weights(self) -> None:
        """Update genre weights based on popularity and ratings"""
        genre_scores = defaultdict(list)
        
        for anime in self.knowledge_base.anime_data.values():
            score = anime.score
            popularity = anime.popularity
            
            # Combine score and popularity for weight calculation
            weight = (score * 0.7 + (10 - np.log10(popularity + 1)) * 0.3)
            
            for genre in anime.genres:
                genre_scores[genre].append(weight)
        
        # Calculate average weight for each genre
        for genre, scores in genre_scores.items():
            self.genre_weights[genre] = np.mean(scores)
    
    def get_recommendations(self, 
                          preferred_genres: List[str] = None,
                          watched_anime: List[str] = None,
                          limit: int = 5) -> List[Dict]:
        """
        Get personalized recommendations based on user preferences
        
        Args:
            preferred_genres: List of genres user prefers
            watched_anime: List of anime titles user has watched
            limit: Number of recommendations to return
            
        Returns:
            List of recommended anime with their details
        """
        scores = {}
        watched_ids = set()
        
        if watched_anime is None:
            watched_anime = []
        if preferred_genres is None:
            preferred_genres = []
        
        # Find IDs of watched anime
        for title in watched_anime:
            for anime_id, anime in self.knowledge_base.anime_data.items():
                if anime.title.lower() == title.lower():
                    watched_ids.add(anime_id)
                    break
        
        # Calculate scores for each anime
        for anime_id, anime in self.knowledge_base.anime_data.items():
            if anime_id in watched_ids:
                continue
                
            score = 0
            
            # Base score from MAL rating
            score += anime.score * 0.4
            
            # Genre preference score
            genre_match_score = 0
            for genre in anime.genres:
                if genre in preferred_genres:
                    genre_match_score += self.genre_weights[genre]
            score += genre_match_score * 0.4
            
            # Popularity bonus (small effect)
            score += (10 - np.log10(anime.popularity + 1)) * 0.2
            
            scores[anime_id] = score
        
        # Get top recommendations
        top_anime_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:limit]
        recommendations = []
        
        for anime_id in top_anime_ids:
            anime = self.knowledge_base.anime_data[anime_id]
            recommendations.append({
                'id': anime_id,
                'title': anime.title,
                'score': anime.score,
                'genres': anime.genres,
                'popularity': anime.popularity,
                'synopsis': anime.synopsis[:200] + '...' if anime.synopsis else '',
                'recommendation_score': scores[anime_id]
            })
        
        return recommendations
    
    def get_similar_anime(self, anime_title: str, limit: int = 5) -> List[Dict]:
        """
        Find similar anime based on genres and other features
        
        Args:
            anime_title: Title of the anime to find similarities for
            limit: Number of similar anime to return
            
        Returns:
            List of similar anime with their details
        """
        # Find the input anime
        target_anime = None
        for anime in self.knowledge_base.anime_data.values():
            if anime.title.lower() == anime_title.lower():
                target_anime = anime
                break
        
        if not target_anime:
            return []
        
        similarities = {}
        target_genres = set(target_anime.genres)
        
        for anime_id, anime in self.knowledge_base.anime_data.items():
            if anime.title == anime_title:
                continue
            
            # Calculate similarity score
            current_genres = set(anime.genres)
            genre_similarity = len(target_genres & current_genres) / len(target_genres | current_genres)
            
            # Score similarity (normalized)
            score_diff = 1 - abs(target_anime.score - anime.score) / 10
            
            # Combine scores
            similarity = (genre_similarity * 0.7 + score_diff * 0.3)
            similarities[anime_id] = similarity
        
        # Get top similar anime
        top_similar = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)[:limit]
        similar_anime = []
        
        for anime_id in top_similar:
            anime = self.knowledge_base.anime_data[anime_id]
            similar_anime.append({
                'id': anime_id,
                'title': anime.title,
                'score': anime.score,
                'genres': anime.genres,
                'similarity_score': similarities[anime_id],
                'common_genres': list(target_genres & set(anime.genres))
            })
        
        return similar_anime
    
    def get_seasonal_recommendations(self, 
                                   season: str = None, 
                                   year: int = None, 
                                   limit: int = 5) -> List[Dict]:
        """
        Get recommendations for anime from a specific season
        
        Args:
            season: Season to filter by (spring/summer/fall/winter)
            year: Year to filter by
            limit: Number of recommendations to return
            
        Returns:
            List of recommended seasonal anime
        """
        seasonal_anime = []
        
        for anime in self.knowledge_base.anime_data.values():
            matches_season = True if season is None else anime.season.lower() == season.lower()
            matches_year = True if year is None else anime.year == year
            
            if matches_season and matches_year:
                seasonal_anime.append({
                    'id': anime.id,
                    'title': anime.title,
                    'score': anime.score,
                    'season': anime.season,
                    'year': anime.year,
                    'genres': anime.genres
                })
        
        # Sort by score and return top ones
        return sorted(seasonal_anime, key=lambda x: x['score'], reverse=True)[:limit]