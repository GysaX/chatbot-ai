import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import requests
import json
import re
import random
from typing import List, Dict, Optional
from collections import defaultdict
import time

class MALDataFetcher:
    """Handles data fetching from MyAnimeList API"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.base_url = "https://api.myanimelist.net/v2"
        self.headers = {
            "X-MAL-CLIENT-ID": client_id
        }
        self.cache = {}
        
    def fetch_anime_list(self, ranking_type: str = "all", limit: int = 100) -> List[Dict]:
        """Fetch anime list from MAL ranking"""
        cache_key = f"ranking_{ranking_type}_{limit}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        url = f"{self.base_url}/anime/ranking"
        params = {
            "ranking_type": ranking_type,
            "limit": limit,
            "fields": "id,title,mean,rank,popularity,num_episodes,genres,synopsis,studios"
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        data = response.json()
        self.cache[cache_key] = data["data"]
        return data["data"]
    
    def fetch_anime_details(self, anime_id: int) -> Dict:
        """Fetch detailed information about a specific anime"""
        if anime_id in self.cache:
            return self.cache[anime_id]
            
        url = f"{self.base_url}/anime/{anime_id}"
        params = {
            "fields": "id,title,mean,rank,popularity,num_episodes,genres,synopsis,studios,related_anime"
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        data = response.json()
        self.cache[anime_id] = data
        return data

class AnimeKnowledgeBase:
    """Manages anime knowledge and provides recommendation logic"""
    
    def __init__(self, mal_fetcher: MALDataFetcher):
        self.mal_fetcher = mal_fetcher
        self.anime_data = {}
        self.genre_index = defaultdict(list)
        self.studio_index = defaultdict(list)
        self.initialize_knowledge_base()
        
    def initialize_knowledge_base(self):
        """Initialize the knowledge base with MAL data"""
        anime_list = self.mal_fetcher.fetch_anime_list(limit=200)
        
        for anime in anime_list:
            anime_node = anime["node"]
            anime_id = anime_node["id"]
            
            self.anime_data[anime_id] = {
                "id": anime_id,
                "title": anime_node["title"],
                "score": anime_node.get("mean", 0),
                "rank": anime_node.get("rank", 0),
                "popularity": anime_node.get("popularity", 0),
                "episodes": anime_node.get("num_episodes", 0),
                "genres": [g["name"] for g in anime_node.get("genres", [])],
                "synopsis": anime_node.get("synopsis", ""),
                "studios": [s["name"] for s in anime_node.get("studios", [])]
            }
            
            # Build indices
            for genre in self.anime_data[anime_id]["genres"]:
                self.genre_index[genre].append(anime_id)
            
            for studio in self.anime_data[anime_id]["studios"]:
                self.studio_index[studio].append(anime_id)
    
    def get_recommendations(self, genres: Optional[List[str]] = None, 
                          min_score: float = 7.0, 
                          limit: int = 5) -> List[Dict]:
        """Get anime recommendations based on criteria"""
        candidates = set()
        
        if genres:
            # Get anime IDs that match all specified genres
            genre_matches = [set(self.genre_index[genre]) for genre in genres]
            if genre_matches:
                candidates = set.intersection(*genre_matches)
        else:
            candidates = set(self.anime_data.keys())
        
        # Filter and sort recommendations
        recommendations = [
            self.anime_data[anime_id] for anime_id in candidates
            if self.anime_data[anime_id]["score"] >= min_score
        ]
        
        recommendations.sort(key=lambda x: (-x["score"], -x["popularity"]))
        return recommendations[:limit]

class AnimeDataset(Dataset):
    """Custom dataset for training the anime chatbot"""
    
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_length: int = 100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to sequence of indices
        sequence = [self.vocab.get(word, self.vocab["<UNK>"]) 
                   for word in text.split()]
        sequence = sequence[:self.max_length]
        sequence = sequence + [self.vocab["<PAD>"]] * (self.max_length - len(sequence))
        
        return torch.tensor(sequence), torch.tensor(label)

class AnimeChatbotModel(nn.Module):
    """Neural network model for intent classification"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_layers: int, num_classes: int, dropout: float = 0.3):
        super(AnimeChatbotModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        output = self.fc(self.dropout(context_vector))
        return output

class AnimeChatbot:
    """Main chatbot class that combines all components"""
    
    def __init__(self, mal_client_id: str):
        self.mal_fetcher = MALDataFetcher(mal_client_id)
        self.knowledge_base = AnimeKnowledgeBase(self.mal_fetcher)
        self.model = None
        self.vocab = None
        self.intent_map = {
            0: "greeting",
            1: "recommendation",
            2: "anime_info",
            3: "genre_inquiry",
            4: "studio_inquiry"
        }
        
        self.response_templates = {
            "greeting": [
                "Konnichiwa! Apa yang ingin kamu ketahui tentang anime?",
                "Ohayou! Ada yang bisa saya bantu tentang anime?",
                "Yoroshiku! Mau rekomendasi anime atau informasi tertentu?"
            ],
            "recommendation": [
                "Berdasarkan kriteria tersebut, saya merekomendasikan {title} (Rating: {score}). {synopsis}",
                "Kamu mungkin akan suka {title}! Anime ini mendapat rating {score} dan {synopsis}",
            ],
            "anime_info": [
                "Anime {title} memiliki rating {score} di MAL. {synopsis}",
                "Info tentang {title}: Rating {score}, Genre: {genres}. {synopsis}"
            ],
            "genre_inquiry": [
                "Untuk genre {genre}, beberapa anime terbaik adalah: {recommendations}",
                "Di genre {genre}, saya merekomendasikan: {recommendations}"
            ],
            "studio_inquiry": [
                "Studio {studio} terkenal dengan anime-anime seperti: {anime_list}",
                "Beberapa karya terbaik dari studio {studio}: {anime_list}"
            ]
        }
    
    def train(self, texts: List[str], labels: List[int], 
              epochs: int = 10, batch_size: int = 32):
        """Train the chatbot model"""
        # Build vocabulary
        words = set()
        for text in texts:
            words.update(text.lower().split())
        
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.vocab.update({word: idx + 2 for idx, word in enumerate(words)})
        
        # Create dataset and dataloader
        dataset = AnimeDataset(texts, labels, self.vocab)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        self.model = AnimeChatbotModel(
            vocab_size=len(self.vocab),
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            num_classes=len(self.intent_map)
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_texts, batch_labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}')
    
    def get_response(self, text: str) -> str:
        """Generate response based on user input"""
        if not self.model:
            return "Model belum ditraining! Silakan training terlebih dahulu."
        
        self.model.eval()
        processed_text = text.lower()
        sequence = [self.vocab.get(word, self.vocab["<UNK>"]) 
                   for word in processed_text.split()]
        
        with torch.no_grad():
            input_tensor = torch.tensor([sequence])
            output = self.model(input_tensor)
            intent = self.intent_map[torch.argmax(output, dim=1).item()]
            
            if intent == "greeting":
                return random.choice(self.response_templates["greeting"])
            
            elif intent == "recommendation":
                recommendations = self.knowledge_base.get_recommendations(limit=1)[0]
                return random.choice(self.response_templates["recommendation"]).format(
                    title=recommendations["title"],
                    score=recommendations["score"],
                    synopsis=recommendations["synopsis"][:100] + "..."
                )
            
            elif intent == "genre_inquiry":
                # Extract genre from text if possible
                genres = [genre for genre in self.knowledge_base.genre_index.keys() 
                         if genre.lower() in processed_text]
                if not genres:
                    genres = [random.choice(list(self.knowledge_base.genre_index.keys()))]
                
                recommendations = self.knowledge_base.get_recommendations(genres=genres, limit=3)
                rec_titles = [f"{anime['title']} (Rating: {anime['score']})" 
                            for anime in recommendations]
                
                return random.choice(self.response_templates["genre_inquiry"]).format(
                    genre=genres[0],
                    recommendations=", ".join(rec_titles)
                )
            
            elif intent == "studio_inquiry":
                # Default to a popular studio if none mentioned
                studio = "MAPPA"  # Example default
                recommendations = [self.anime_data[anime_id] 
                                 for anime_id in self.knowledge_base.studio_index[studio][:3]]
                
                return random.choice(self.response_templates["studio_inquiry"]).format(
                    studio=studio,
                    anime_list=", ".join([f"{anime['title']} (Rating: {anime['score']})" 
                                        for anime in recommendations])
                )
            
            else:
                return "Maaf, saya tidak mengerti pertanyaan Anda. Bisa tolong diulangi?"

# Example usage
def main():
    # Initialize chatbot with MAL client ID
    mal_client_id = "your_mal_client_id_here"
    chatbot = AnimeChatbot(mal_client_id)
    
    # Training data
    training_data = [
        ("halo apa kabar", 0),
        ("recommend anime dong", 1),
        ("ceritakan tentang one piece", 2),
        ("anime action bagus apa", 3),
        ("anime dari studio mappa", 4),
        # Add more training examples...
    ]
    
    texts, labels = zip(*training_data)
    chatbot.train(texts, labels)
    
    # Test the chatbot
    test_questions = [
        "halo!",
        "recommend anime dong",
        "anime action apa yang bagus?",
        "anime dari studio mappa apa aja?"
    ]
    
    for question in test_questions:
        response = chatbot.get_response(question)
        print(f"Q: {question}")
        print(f"A: {response}\n")

if __name__ == "__main__":
    main()