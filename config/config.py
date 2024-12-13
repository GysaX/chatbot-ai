# MyAnimeList API Configuration
MAL_CLIENT_ID = "da48606ea380c846af7adbfff9fcd008"  # Replace with your MAL Client ID

API_CONFIG = {
    "base_url": "https://api.myanimelist.net/v2",
    "cache_timeout": 3600,  # 1 hour in seconds
    "request_timeout": 10,  # 10 seconds timeout for requests
    "max_retries": 3
}

# Model Configuration
MODEL_CONFIG = {
    "embedding_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "max_sequence_length": 100,
    "vocab_size": 10000  # Will be updated based on training data
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "early_stopping_patience": 3
}

# Data Configuration
DATA_CONFIG = {
    "min_word_frequency": 2,  # Minimum frequency for word to be included in vocabulary
    "max_anime_fetch": 500,   # Maximum number of anime to fetch from MAL
    "cache_dir": "cache",     # Directory for caching API responses
    "seed": 42               # Random seed for reproducibility
}

# Intent Configuration
INTENT_MAPPING = {
    0: "greeting",
    1: "recommendation",
    2: "anime_info",
    3: "genre_inquiry",
    4: "studio_inquiry"
}

# Response Templates
RESPONSE_TEMPLATES = {
    "greeting": [
        "Konnichiwa! Apa yang ingin kamu ketahui tentang anime?",
        "Ohayou! Ada yang bisa saya bantu tentang anime?",
        "Yoroshiku! Mau rekomendasi anime atau informasi tertentu?"
    ],
    "recommendation": [
        "Berdasarkan kriteria tersebut, saya merekomendasikan {title} (Rating: {score}). {synopsis}",
        "Kamu mungkin akan suka {title}! Anime ini mendapat rating {score} dan {synopsis}"
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