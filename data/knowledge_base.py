from typing import Dict, List, Optional, Set
from collections import defaultdict
import json
import os
from datetime import datetime, timedelta

class AnimeData:
    """Class to represent individual anime entries"""
    
    def __init__(self, data: Dict):
        self.id = data.get("id")
        self.title = data.get("title", "Unknown Title")
        self.english_title = data.get("english_title", "")
        
        # Handle score with proper conversion
        try:
            self.score = float(data.get("mean", 0.0))
        except (TypeError, ValueError):
            self.score = 0.0
        
        # Handle numeric fields
        try:
            self.popularity = int(data.get("popularity", 0))
            self.members = int(data.get("members", 0))
            self.favorites = int(data.get("favorites", 0))
            self.episodes = int(data.get("num_episodes", 0))
        except (TypeError, ValueError):
            self.popularity = 0
            self.members = 0
            self.favorites = 0
            self.episodes = 0
        
        # Handle genres
        self.genres = []
        genres_data = data.get("genres", [])
        if isinstance(genres_data, list):
            self.genres = [g.get("name", "") for g in genres_data if isinstance(g, dict)]
        
        # Handle studios
        self.studios = []
        studios_data = data.get("studios", [])
        if isinstance(studios_data, list):
            self.studios = [s.get("name", "") for s in studios_data if isinstance(s, dict)]
        
        self.synopsis = data.get("synopsis", "No synopsis available.")
        self.aired = data.get("aired", {})
        
        try:
            self.year = int(data.get("year", 0)) if data.get("year") else None
        except (TypeError, ValueError):
            self.year = None
            
        self.season = data.get("season", "")

    def to_dict(self) -> Dict:
        """Convert anime data to dictionary format"""
        return {
            "id": self.id,
            "title": self.title,
            "english_title": self.english_title,
            "mean": self.score,  # Use 'mean' instead of 'score' for consistency
            "popularity": self.popularity,
            "members": self.members,
            "favorites": self.favorites,
            "genres": [{"name": genre} for genre in self.genres],  # Format genres properly
            "studios": [{"name": studio} for studio in self.studios],  # Format studios properly
            "synopsis": self.synopsis,
            "num_episodes": self.episodes,  # Use 'num_episodes' instead of 'episodes'
            "aired": self.aired,
            "year": self.year,
            "season": self.season
        }
    
    def _standardize_genre(self, genre: str) -> str:
        """Standardize genre names to match common formats"""
        genre_mapping = {
            "shounen": "Shounen",
            "shonen": "Shounen",
            "shoujo": "Shoujo",
            "shojo": "Shoujo",
            "sci fi": "Sci-Fi",
            "sci-fi": "Sci-Fi",
            "slice of life": "Slice of Life",
            "martial arts": "Martial Arts",
            # Add more mappings as needed
        }
        
        genre = genre.strip()
        # Check direct mapping
        if genre.lower() in genre_mapping:
            return genre_mapping[genre.lower()]
        # Capitalize first letter of each word
        return genre.title()
    
    def is_valid(self) -> bool:
        """Check if anime data is valid and complete"""
        return (
            self.id is not None and
            self.title and
            isinstance(self.genres, list) and
            len(self.genres) > 0 and
            self.score >= 0 and
            self.episodes >= 0
        )

class AnimeKnowledgeBase:
    """Manages anime data with improved validation and filtering"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.anime_data: Dict[int, AnimeData] = {}
        self.genre_index: Dict[str, Set[int]] = defaultdict(set)
        self.studio_index: Dict[str, Set[int]] = defaultdict(set)
        self.year_index: Dict[int, Set[int]] = defaultdict(set)
        self.season_index: Dict[str, Set[int]] = defaultdict(set)
        
        # Track valid genres
        self.valid_genres = set()
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def add_anime(self, data: Dict) -> bool:
        """
        Add new anime to the knowledge base with validation
        Returns True if successfully added, False otherwise
        """
        try:
            anime = AnimeData(data)
            
            # Validate anime data
            if not anime.is_valid():
                return False
            
            anime_id = anime.id
            if anime_id is None:
                return False
            
            self.anime_data[anime_id] = anime
            
            # Update indices
            for genre in anime.genres:
                if isinstance(genre, str):
                    self.genre_index[genre].add(anime_id)
                    self.valid_genres.add(genre)
            
            for studio in anime.studios:
                if isinstance(studio, str):
                    self.studio_index[studio].add(anime_id)
            
            if anime.year:
                self.year_index[anime.year].add(anime_id)
            
            if anime.season:
                self.season_index[anime.season].add(anime_id)
            
            return True
            
        except Exception as e:
            print(f"Error adding anime: {str(e)}")
            return False
    
    def get_anime_by_genre(self, genre: str, limit: int = 10) -> List[AnimeData]:
        """Get top anime in a specific genre with better handling"""
        # Standardize genre name
        genre = AnimeData._standardize_genre(genre)
        
        # Get anime IDs for this genre
        anime_ids = self.genre_index.get(genre, set())
        if not anime_ids:
            # Try case-insensitive search
            for g in self.genre_index:
                if g.lower() == genre.lower():
                    anime_ids = self.genre_index[g]
                    break
        
        # Get anime objects
        anime_list = []
        for aid in anime_ids:
            anime = self.anime_data.get(aid)
            if anime and anime.score > 0:  # Only include anime with valid scores
                anime_list.append(anime)
        
        # Sort by score and return top N
        return sorted(anime_list, key=lambda x: x.score, reverse=True)[:limit]
    
    def get_available_genres(self) -> List[str]:
        """Get list of all available genres"""
        return sorted(list(self.valid_genres))
    
    def get_genre_stats(self) -> Dict[str, Dict]:
        """Get statistics for each genre"""
        stats = {}
        for genre in self.valid_genres:
            anime_list = self.get_anime_by_genre(genre)
            if anime_list:
                avg_score = sum(a.score for a in anime_list) / len(anime_list)
                stats[genre] = {
                    "count": len(self.genre_index[genre]),
                    "avg_score": round(avg_score, 2),
                    "top_anime": anime_list[0].title if anime_list else None
                }
        return stats

    def print_genre_summary(self) -> None:
        """Print summary of available genres and their statistics"""
        stats = self.get_genre_stats()
        print("\nAnime Genre Summary:")
        print("-" * 50)
        for genre, data in stats.items():
            print(f"\nGenre: {genre}")
            print(f"Number of anime: {data['count']}")
            print(f"Average score: {data['avg_score']}")
            print(f"Top anime: {data['top_anime']}")

    def save_to_cache(self) -> None:
        """Save knowledge base to cache"""
        try:
            cache_file = os.path.join(self.cache_dir, "anime_cache.json")
            
            data = {
                "last_updated": datetime.now().isoformat(),
                "anime_data": {
                    str(aid): anime.to_dict()
                    for aid, anime in self.anime_data.items()
                }
            }
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            print(f"Cache saved successfully to {cache_file}")
        except Exception as e:
            print(f"Error saving cache: {str(e)}")

    def load_from_cache(self) -> bool:
        """Load knowledge base from cache"""
        try:
            cache_file = os.path.join(self.cache_dir, "anime_cache.json")
            
            if not os.path.exists(cache_file):
                print("No cache file found")
                return False
                
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Check if cache is older than 24 hours
            last_updated = datetime.fromisoformat(data["last_updated"])
            if datetime.now() - last_updated > timedelta(hours=24):
                print("Cache is too old, will fetch new data")
                return False
            
            # Load data
            success_count = 0
            for anime_data in data["anime_data"].values():
                if self.add_anime(anime_data):
                    success_count += 1
            
            print(f"Loaded {success_count} anime from cache")
            return success_count > 0
            
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
            return False

    def clear_cache(self) -> None:
        """Clear the cache file"""
        try:
            cache_file = os.path.join(self.cache_dir, "anime_cache.json")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print("Cache cleared successfully")
            else:
                print("No cache file found")
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")