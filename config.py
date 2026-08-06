cat > config.py << 'EOF'
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///attendance.db')
    
    # Face Recognition
    INSIGHTFACE_MODEL = os.getenv('INSIGHTFACE_MODEL', 'buffalo_l')
    FACE_DETECTION_SIZE = (640, 640)
    RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', '0.6'))
    
    # Server
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8000))
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Data Directories
    DATA_DIR = 'data'
    EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
    TEST_IMAGES_DIR = os.path.join(DATA_DIR, 'test_images')
    
    # Create directories if they don't exist
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
EOF