from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from main import AnimeChatbotSystem
from models.dataset import VocabularyBuilder
from config.config import MAL_CLIENT_ID, MODEL_CONFIG, TRAINING_CONFIG
import logging
import os
import torch
from torch.utils.data import DataLoader, random_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
template_dir = os.path.abspath('templates')
static_dir = os.path.abspath('static')
app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)
CORS(app)

def initialize_chatbot():
    """Initialize chatbot with proper model loading"""
    try:
        # Initialize chatbot system
        chatbot = AnimeChatbotSystem(MAL_CLIENT_ID)
        logger.info("Chatbot system initialized")

        # Prepare training data
        texts, labels = chatbot.prepare_training_data()
        logger.info("Training data prepared")

        # Initialize vocabulary builder
        chatbot.vocab_builder = VocabularyBuilder(min_freq=2)
        chatbot.vocab_builder.build_vocab(texts)
        logger.info("Vocabulary built")

        # Create dataset
        dataset = chatbot.create_dataset(texts, labels)
        logger.info("Dataset created")

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
        num_classes = len(set(labels))
        chatbot.initialize_model(num_classes)
        logger.info("Model initialized")

        # Load or train model
        model_path = 'best_model.pth'
        if os.path.exists(model_path):
            chatbot.load_model(model_path)
            logger.info("Model loaded from file")
        else:
            logger.info("Training new model...")
            chatbot.train_model(train_loader, val_loader)
            logger.info("Model training completed")

        return chatbot

    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        raise

# Initialize chatbot globally
try:
    chatbot = initialize_chatbot()
    logger.info("Chatbot initialized successfully")
except Exception as e:
    logger.error(f"Error initializing chatbot: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Ensure model is in eval mode
        chatbot.model.eval()
        
        # Get response from chatbot
        intent = chatbot.predict_intent(message)
        response = chatbot.generate_response(intent, message, 'web_user')
        
        logger.info(f"User message: {message}")
        logger.info(f"Bot response: {response}")
        
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Ensure directories exist
        os.makedirs('templates', exist_ok=True)
        os.makedirs('static', exist_ok=True)
        
        logger.info("Starting Flask server on port 5000...")
        app.run(host='127.0.0.1', port=5000, debug=False)  # Set debug=False to prevent double initialization
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")