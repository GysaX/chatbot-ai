import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import os
import json
from typing import List, Tuple, Dict
import logging
from datetime import datetime

# Import custom modules
from config.config import MODEL_CONFIG, TRAINING_CONFIG, MAL_CLIENT_ID
from models.model import AnimeChatbotModel
from models.dataset import AnimeDataset, VocabularyBuilder
from data.knowledge_base import AnimeKnowledgeBase
from data.mal_fetcher import MALDataFetcher
from utils.training import AnimeTrainingDataGenerator
from utils.recommender import AnimeRecommender
from utils.sentiment import AnimeSentimentAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnimeChatbotSystem:
    """
    Main system class that integrates all components of the anime chatbot
    """
    def __init__(self, mal_client_id: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.mal_fetcher = MALDataFetcher(mal_client_id)
        self.knowledge_base = AnimeKnowledgeBase()
        self.recommender = AnimeRecommender(self.knowledge_base)
        self.sentiment_analyzer = AnimeSentimentAnalyzer()
        
        # Model components
        self.vocab_builder = None
        self.model = None
        
        # Initialize knowledge base
        self._initialize_knowledge_base()
        
    def _initialize_knowledge_base(self) -> None:
        """Initialize or load knowledge base from cache"""
        if not self.knowledge_base.load_from_cache():
            logger.info("Fetching new data from MyAnimeList...")
            anime_list = self.mal_fetcher.fetch_anime_list(limit=200)
            for anime in anime_list:
                self.knowledge_base.add_anime(anime["node"])
            self.knowledge_base.save_to_cache()
        logger.info("Knowledge base initialized successfully")
    
    def prepare_training_data(self) -> Tuple[List[str], List[int]]:
        """Prepare training data for the model"""
        training_generator = AnimeTrainingDataGenerator(self.knowledge_base)
        training_data = training_generator.generate_training_data(num_samples=1000)
        texts, labels = zip(*training_data)
        return list(texts), list(labels)
    
    def create_dataset(self, texts: List[str], labels: List[int]) -> AnimeDataset:
        """
        Create dataset from texts and labels
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            
        Returns:
            AnimeDataset instance
        """
        if self.vocab_builder is None:
            self.vocab_builder = VocabularyBuilder(min_freq=2)
            self.vocab_builder.build_vocab(texts)
            logger.info("Vocabulary builder initialized and vocabulary built")

        dataset = AnimeDataset(
            texts=texts,
            labels=labels,
            vocab_builder=self.vocab_builder,
            max_length=MODEL_CONFIG["max_sequence_length"]
        )
        
        logger.info(f"Created dataset with {len(texts)} samples")
        return dataset
    
    def initialize_model(self, num_classes: int) -> None:
        """Initialize the model with proper configuration"""
        self.model = AnimeChatbotModel(
            vocab_size=self.vocab_builder.vocab_size,
            embedding_dim=MODEL_CONFIG["embedding_dim"],
            hidden_dim=MODEL_CONFIG["hidden_dim"],
            num_classes=num_classes,
            num_encoder_layers=MODEL_CONFIG.get("num_encoder_layers", 6),
            num_heads=MODEL_CONFIG.get("num_heads", 8),
            dropout=MODEL_CONFIG["dropout"],
            max_seq_length=MODEL_CONFIG["max_sequence_length"]
        ).to(self.device)
        logger.info("Model initialized successfully")
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train the model"""
        logger.info("Starting model training...")
        
        optimizer = self.model.configure_optimizers(
            lr=TRAINING_CONFIG["learning_rate"]
        )
        
        best_val_loss = float('inf')
        patience = TRAINING_CONFIG.get("patience", 3)
        patience_counter = 0
        
        for epoch in range(TRAINING_CONFIG["num_epochs"]):
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            for batch_idx, (texts, labels) in enumerate(train_loader):
                texts, labels = texts.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                loss = self.model.training_step((texts, labels))
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{TRAINING_CONFIG['num_epochs']} - "
                              f"Batch {batch_idx} - Loss: {loss.item():.4f}")
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for texts, labels in val_loader:
                    texts, labels = texts.to(self.device), labels.to(self.device)
                    loss = self.model.training_step((texts, labels))
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            logger.info(f"Epoch {epoch+1} completed - "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break
    
    def save_model(self, filename: str) -> None:
        """Save model and vocabulary"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'vocab_builder': self.vocab_builder,
            'model_config': MODEL_CONFIG
        }
        torch.save(save_dict, filename)
        logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename: str) -> None:
        """Load model and vocabulary with error handling"""
        try:
            if os.path.exists(filename):
                save_dict = torch.load(filename, map_location=self.device)
                self.vocab_builder = save_dict['vocab_builder']
                
                # Check if number of classes matches
                saved_num_classes = save_dict['model_state_dict']['classifier.3.bias'].size(0)
                current_num_classes = 5  # We now use 5 fixed classes
                
                # Initialize model
                self.initialize_model(current_num_classes)
                
                if saved_num_classes != current_num_classes:
                    logger.warning(f"Saved model has {saved_num_classes} classes but we need {current_num_classes} classes.")
                    logger.warning("Training new model...")
                    return False
                
                # Load state dict if number of classes matches
                self.model.load_state_dict(save_dict['model_state_dict'])
                logger.info(f"Model loaded from {filename}")
                return True
            else:
                logger.info(f"No model file found at {filename}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def predict_intent(self, text: str) -> int:
        """Predict intent from text input with improved detection"""
        text_lower = text.lower()
        
        # Keywords untuk setiap intent
        intent_keywords = {
            "genre": [
                "genre", "kategori", "tipe", "jenis",
                "action", "adventure", "comedy", "drama", "fantasy",
                "horror", "mystery", "romance", "sci-fi", "slice of life",
                "sports", "supernatural", "thriller", "mecha", "music",
                "psychological", "seinen", "shounen", "shoujo"
            ],
            "info": [
                "ceritakan tentang", "sinopsis", "cerita", "plot",
                "info", "informasi", "tentang anime", "apa itu",
                "bagaimana cerita", "seperti apa"
            ],
            "recommendation": [
                "recommend", "rekomendasi", "rekomendasikan",
                "sarankan", "saran", "bagus", "terbaik",
                "populer", "rating tinggi", "anime apa"
            ],
            "greeting": [
                "halo", "hi", "hey", "ohayou", "konnichiwa",
                "selamat", "salam"
            ]
        }
        
        # Cek keyword dalam text
        if any(keyword in text_lower for keyword in intent_keywords["info"]):
            return 2  # anime_info intent
        elif any(keyword in text_lower for keyword in intent_keywords["genre"]):
            return 3  # genre inquiry intent
        elif any(keyword in text_lower for keyword in intent_keywords["recommendation"]):
            return 1  # recommendation intent
        elif any(keyword in text_lower for keyword in intent_keywords["greeting"]):
            return 0  # greeting intent
        
        # Jika tidak ada keyword yang cocok, gunakan model prediction
        self.model.eval()
        sequence = self.vocab_builder.encode(text, MODEL_CONFIG["max_sequence_length"])
        input_tensor = torch.tensor([sequence]).to(self.device)
        
        with torch.no_grad():
            predictions = self.model.predict(input_tensor)
            predicted_class = torch.argmax(predictions, dim=1).item()
        
        return predicted_class

    
    def generate_response(self, intent: int, text: str, user_id: str) -> str:
        """Generate response based on predicted intent with improved genre detection"""
        # Check for genre-related queries first
        genre_keywords = ["genre", "kategori", "tipe", "jenis"]
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in genre_keywords):
            return self._handle_genre_inquiry(text, user_id)
        
        # Normal intent handling
        intent_map = {
            0: self._handle_greeting,
            1: self._handle_recommendation,
            2: self._handle_anime_info,
            3: self._handle_genre_inquiry,
            4: self._handle_studio_inquiry
        }
        
        handler = intent_map.get(intent, self._handle_unknown)
        return handler(text, user_id)
    
    def _handle_greeting(self, text: str, user_id: str) -> str:
        return "Konnichiwa! Apa yang ingin kamu ketahui tentang anime hari ini?"
    
    def _handle_recommendation(self, text: str, user_id: str) -> str:
        """
        Handle recommendation requests with better genre detection and response formatting
        """
        # List of all possible genres for detection
        common_genres = [
            "Action", "Adventure", "Comedy", "Drama", "Fantasy", 
            "Horror", "Mystery", "Romance", "Sci-Fi", "Slice of Life",
            "Sports", "Supernatural", "Thriller", "Mecha", "Music",
            "Psychological", "Seinen", "Shounen", "Shoujo"
        ]
        
        # Detect genre from user input
        requested_genre = None
        text_lower = text.lower()
        for genre in common_genres:
            if genre.lower() in text_lower:
                requested_genre = genre
                break
        
        try:
            if requested_genre:
                # Get recommendations for specific genre
                recommendations = self.knowledge_base.get_anime_by_genre(requested_genre, limit=5)
                response = f"Untuk genre {requested_genre}, saya merekomendasikan:\n\n"
            else:
                # Get general recommendations based on rating
                all_anime = list(self.knowledge_base.anime_data.values())
                recommendations = sorted(all_anime, key=lambda x: x.score, reverse=True)[:5]
                response = "Berikut rekomendasi anime dengan rating tertinggi:\n\n"
            
            if not recommendations:
                return f"Maaf, saya tidak menemukan rekomendasi anime{' untuk genre ' + requested_genre if requested_genre else ''} saat ini."
            
            # Format each recommendation with detailed information
            for i, anime in enumerate(recommendations, 1):
                genres = ", ".join(anime.genres[:3])  # Show only first 3 genres
                response += (
                    f"{i}. {anime.title}\n"
                    f"   Rating: {anime.score:.2f}/10\n"
                    f"   Genre: {genres}\n"
                    f"   Episodes: {anime.episodes}\n"
                    f"   {anime.synopsis[:150]}...\n\n"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in recommendation handler: {str(e)}")
            return "Maaf, terjadi kesalahan saat mengambil rekomendasi. Mohon coba lagi."

    def _handle_anime_info(self, text: str, user_id: str) -> str:
        """
        Handle anime information requests with improved detection
        """
        # Get all anime titles and create fuzzy matching
        anime_titles = {anime.title.lower(): anime 
                    for anime in self.knowledge_base.anime_data.values()}
        
        # Extract potential anime title from text
        text_lower = text.lower()
        found_anime = None
        
        # Remove common words that might interfere with title matching
        common_words = ["ceritakan", "tentang", "anime", "sinopsis", "info", "informasi"]
        for word in common_words:
            text_lower = text_lower.replace(word, "")
        
        text_lower = text_lower.strip()
        
        # Try to find the longest matching title
        max_length = 0
        for title, anime in anime_titles.items():
            if title in text_lower and len(title) > max_length:
                found_anime = anime
                max_length = len(title)
        
        if not found_anime:
            return "Maaf, saya tidak menemukan anime yang Anda maksud. Mohon sebutkan judul anime dengan lebih spesifik."
        
        try:
            # Format detailed anime information
            genres = ", ".join(found_anime.genres)
            studios = ", ".join(found_anime.studios) if found_anime.studios else "Tidak diketahui"
            
            response = (
                f"Informasi tentang {found_anime.title}:\n\n"
                f"Rating: {found_anime.score:.2f}/10\n"
                f"Peringkat Popularitas: #{found_anime.popularity}\n"
                f"Episode: {found_anime.episodes}\n"
                f"Genre: {genres}\n"
                f"Studio: {studios}\n\n"
                f"Sinopsis:\n{found_anime.synopsis}\n"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in anime info handler: {str(e)}")
            return "Maaf, terjadi kesalahan saat mengambil informasi anime. Mohon coba lagi."
    
    
    def _handle_genre_inquiry(self, text: str, user_id: str) -> str:
        """
        Handle genre-specific inquiries with improved response
        """
        # List of all possible genres
        common_genres = {
            "action": "Action",
            "adventure": "Adventure",
            "comedy": "Comedy",
            "drama": "Drama",
            "fantasy": "Fantasy",
            "horror": "Horror",
            "mystery": "Mystery",
            "romance": "Romance",
            "sci-fi": "Sci-Fi",
            "slice of life": "Slice of Life",
            "sports": "Sports",
            "supernatural": "Supernatural",
            "thriller": "Thriller",
            "mecha": "Mecha",
            "music": "Music",
            "psychological": "Psychological",
            "seinen": "Seinen",
            "shounen": "Shounen",
            "shoujo": "Shoujo"
        }
        
        # Detect genre from user input
        text_lower = text.lower()
        requested_genre = None
        
        for genre_key, genre_value in common_genres.items():
            if genre_key in text_lower:
                requested_genre = genre_value
                break
        
        if not requested_genre:
            genres_list = ", ".join(common_genres.values())
            return f"Mohon sebutkan genre anime yang ingin Anda cari. Genre yang tersedia:\n{genres_list}"
        
        try:
            # Get all anime and filter by genre
            all_anime = list(self.knowledge_base.anime_data.values())
            genre_anime = [
                anime for anime in all_anime
                if requested_genre in anime.genres
            ]
            
            # Sort by score
            genre_anime.sort(key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
            # Take top 5
            genre_anime = genre_anime[:5]
            
            if not genre_anime:
                return f"Maaf, saya tidak menemukan anime untuk genre {requested_genre}."
            
            response = f"Top 5 anime dengan genre {requested_genre}:\n\n"
            
            for i, anime in enumerate(genre_anime, 1):
                other_genres = [g for g in anime.genres if g != requested_genre]
                other_genres_str = ", ".join(other_genres[:2]) if other_genres else "Tidak ada"
                
                response += (
                    f"{i}. {anime.title}\n"
                    f"   Rating: {anime.score:.2f}/10\n"
                    f"   Genre: {requested_genre}, {other_genres_str}\n"
                    f"   Episodes: {anime.episodes}\n"
                    f"   Sinopsis: {anime.synopsis[:150]}...\n\n"
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in genre inquiry handler: {str(e)}")
            return "Maaf, terjadi kesalahan saat mencari anime. Mohon coba lagi."
    
    def _handle_studio_inquiry(self, text: str, user_id: str) -> str:
        # Simplified studio handling
        return "Studio tersebut telah memproduksi beberapa anime populer seperti..."
    
    def _handle_unknown(self, text: str, user_id: str) -> str:
        return "Maaf, saya tidak mengerti pertanyaan Anda. Bisa tolong diulangi?"

def main():
    """Main execution function"""
    try:
        # Initialize system
        chatbot = AnimeChatbotSystem(MAL_CLIENT_ID)
        
        # Prepare training data
        texts, labels = chatbot.prepare_training_data()
        
        # Initialize vocabulary
        chatbot.vocab_builder = VocabularyBuilder(min_freq=2)
        chatbot.vocab_builder.build_vocab(texts)
        
        # Create dataset
        dataset = AnimeDataset(
            texts=texts,
            labels=labels,
            vocab_builder=chatbot.vocab_builder,
            max_length=MODEL_CONFIG["max_sequence_length"]
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=TRAINING_CONFIG["batch_size"]
        )
        
        # Initialize model
        num_classes = 5  # Fixed number of classes
        chatbot.initialize_model(num_classes)
        
        # Try to load model or train new one
        model_loaded = False
        if os.path.exists('best_model.pth'):
            model_loaded = chatbot.load_model('best_model.pth')
        
        if not model_loaded:
            logger.info("Training new model...")
            chatbot.train_model(train_loader, val_loader)
        
        # Interactive loop
        logger.info("\nChatbot is ready! Type 'quit' to exit.")
        user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'quit':
                break
            
            intent = chatbot.predict_intent(user_input)
            response = chatbot.generate_response(intent, user_input, user_id)
            print(f"Bot: {response}")
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
