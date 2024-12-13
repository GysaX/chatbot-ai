from typing import List, Tuple, Dict
import random
from data.knowledge_base import AnimeKnowledgeBase

class AnimeTrainingDataGenerator:
    """Generates training data for the chatbot"""
    
    def __init__(self, knowledge_base: AnimeKnowledgeBase):
        """
        Initialize the training data generator
        
        Args:
            knowledge_base: Instance of AnimeKnowledgeBase containing anime data
        """
        self.knowledge_base = knowledge_base
        
        # Initialize training patterns
        self.greeting_patterns = [
            "halo",
            "hi",
            "hey",
            "ohayou",
            "konnichiwa",
            "konbanwa",
            "selamat pagi",
            "selamat siang",
            "selamat malam",
            "hai apa kabar",
            "halo bot",
            "hi anime bot"
        ]
        
        self.recommendation_patterns = [
            "recommend anime {genre}",
            "anime {genre} bagus apa",
            "recommend anime seperti {title}",
            "anime bagus untuk ditonton",
            "rekomendasi anime terbaru",
            "anime rating tinggi",
            "anime populer musim ini",
            "recommend me some anime",
            "anime apa yang bagus",
            "mau nonton anime apa ya"
        ]
        
        self.info_patterns = [
            "ceritakan tentang {title}",
            "sinopsis {title}",
            "rating {title} berapa",
            "berapa episode {title}",
            "kapan {title} rilis",
            "studio yang buat {title}",
            "genre {title} apa",
            "{title} anime tentang apa",
            "review {title}",
            "info anime {title}"
        ]
        
        self.genre_patterns = [
            "anime genre {genre} apa saja",
            "rekomendasi anime {genre}",
            "anime {genre} terbaik",
            "anime {genre} terpopuler",
            "genre {genre} bagus apa",
            "top anime {genre}",
            "list anime {genre}",
            "anime {genre} untuk pemula",
            "anime {genre} klasik",
            "anime {genre} terbaru"
        ]
        
        self.studio_patterns = [
            "anime dari studio {studio}",
            "karya studio {studio}",
            "anime produksi {studio}",
            "studio {studio} buat anime apa aja",
            "rekomendasi anime dari {studio}",
            "top anime studio {studio}",
            "anime terbaik dari {studio}",
            "list anime {studio}",
            "anime {studio} terbaru",
            "studio {studio} terkenal karena apa"
        ]

    def collect_anime_data(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Collect anime titles, genres, and studios from knowledge base
        
        Returns:
            Tuple containing lists of titles, genres, and studios
        """
        titles = []
        genres = set()
        studios = set()
        
        for anime in self.knowledge_base.anime_data.values():
            titles.append(anime.title)
            genres.update(anime.genres)
            studios.update(anime.studios)
        
        return titles, list(genres), list(studios)

    def generate_training_data(self, num_samples: int = 1000) -> List[Tuple[str, int]]:
        """
        Generate training data pairs (text, label)
        
        Args:
            num_samples: Number of training samples to generate
            
        Returns:
            List of (text, label) tuples
        """
        training_data = []
        samples_per_category = num_samples // 5
        
        # Get anime data
        titles, genres, studios = self.collect_anime_data()
        
        # Generate greeting samples
        for _ in range(samples_per_category):
            text = random.choice(self.greeting_patterns)
            training_data.append((text, 0))
        
        # Generate recommendation samples
        for _ in range(samples_per_category):
            pattern = random.choice(self.recommendation_patterns)
            if "{genre}" in pattern and genres:
                text = pattern.format(genre=random.choice(genres))
            elif "{title}" in pattern and titles:
                text = pattern.format(title=random.choice(titles))
            else:
                text = pattern
            training_data.append((text, 1))
        
        # Generate info samples
        for _ in range(samples_per_category):
            if titles:
                pattern = random.choice(self.info_patterns)
                text = pattern.format(title=random.choice(titles))
                training_data.append((text, 2))
        
        # Generate genre inquiry samples
        for _ in range(samples_per_category):
            if genres:
                pattern = random.choice(self.genre_patterns)
                text = pattern.format(genre=random.choice(genres))
                training_data.append((text, 3))
        
        # Generate studio inquiry samples
        for _ in range(samples_per_category):
            if studios:
                pattern = random.choice(self.studio_patterns)
                text = pattern.format(studio=random.choice(studios))
                training_data.append((text, 4))
        
        # Shuffle the training data
        random.shuffle(training_data)
        return training_data

    def generate_validation_data(self, num_samples: int = 200) -> List[Tuple[str, int]]:
        """
        Generate validation data
        
        Args:
            num_samples: Number of validation samples to generate
            
        Returns:
            List of (text, label) tuples for validation
        """
        return self.generate_training_data(num_samples)

    def get_intent_labels(self) -> Dict[int, str]:
        """
        Get mapping of intent numbers to labels
        
        Returns:
            Dictionary mapping intent numbers to their string labels
        """
        return {
            0: "greeting",
            1: "recommendation",
            2: "anime_info",
            3: "genre_inquiry",
            4: "studio_inquiry"
        }

    def get_num_intents(self) -> int:
        """
        Get total number of intents
        
        Returns:
            Number of different intents
        """
        return len(self.get_intent_labels())

    def get_intent_distribution(self, training_data: List[Tuple[str, int]]) -> Dict[str, int]:
        """
        Get distribution of intents in training data
        
        Args:
            training_data: List of (text, label) training pairs
            
        Returns:
            Dictionary with count of samples for each intent
        """
        distribution = {label: 0 for label in self.get_intent_labels().values()}
        intent_map = self.get_intent_labels()
        
        for _, label in training_data:
            intent_name = intent_map[label]
            distribution[intent_name] += 1
            
        return distribution

    def print_training_stats(self, training_data: List[Tuple[str, int]]) -> None:
        """
        Print statistics about the training data
        
        Args:
            training_data: List of (text, label) training pairs
        """
        distribution = self.get_intent_distribution(training_data)
        total = len(training_data)
        
        print("\nTraining Data Statistics:")
        print(f"Total samples: {total}")
        print("\nIntent distribution:")
        for intent, count in distribution.items():
            percentage = (count / total) * 100
            print(f"{intent}: {count} samples ({percentage:.1f}%)")
