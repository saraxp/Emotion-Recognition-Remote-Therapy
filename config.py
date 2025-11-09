
"""
Emotion Recognition Configuration
Includes existing settings + new Gemini API configuration
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== EXISTING SETTINGS ====================

# Flask Settings
# Flask Settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true" 
HOST = '0.0.0.0'
PORT = int(os.getenv("PORT", "5000"))  # Read from environment

# Model Settings
# Hugging Face model (public)
HF_REPO_ID = "saraxp/emotion_model_best.h5"  # <user>/<repo>
HF_MODEL_FILENAME = "emotion_model_best.h5"  # file inside the repo
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Session Settings
SESSION_DATA_DIR = 'session_data'
DATA_RETENTION_DAYS = 7

# Video Settings
FRAME_SKIP = 2  # Process every nth frame
EMOTION_WINDOW_SIZE = 10  # Smoothing window

# Cloud Storage (Optional)
USE_CLOUD_STORAGE = False
CLOUD_PROVIDER = 'firebase'  # Options: firebase, mongodb, s3

# ==================== NEW: API KEYS & SECRETS ====================

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"  # Fast and free tier available

# Flask Secret Key (for sessions)
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "")

# Firebase Configuration (if needed)
FIREBASE_CREDENTIALS= os.getenv("FIREBASE_CREDENTIALS", "")

# ==================== VALIDATION ====================

def validate_config():
    """Validate that required configuration is present"""
    issues = []
    warnings = []
    
    # Critical checks
    if not FLASK_SECRET_KEY:
        issues.append("❌ FLASK_SECRET_KEY not set - Security risk!")
    
    # Warning checks
    if not GEMINI_API_KEY:
        warnings.append("⚠️  GEMINI_API_KEY not set - AI summaries will use fallback mode")
    
    if USE_CLOUD_STORAGE and not FIREBASE_CREDENTIALS and CLOUD_PROVIDER == 'firebase':
        warnings.append("⚠️  Firebase credentials not set but cloud storage is enabled")
    
    # Print results
    if issues or warnings:
        print("\n" + "="*70)
        print("CONFIGURATION CHECK:")
        print("="*70)
        
        if issues:
            print("\n❌ CRITICAL ISSUES:")
            for issue in issues:
                print(f"  {issue}")
        
        if warnings:
            print("\n⚠️  WARNINGS:")
            for warning in warnings:
                print(f"  {warning}")
        
        print("\nTo fix: Create a .env file with your configuration")
        print("See .env.template for reference")
        print("="*70 + "\n")
    else:
        print("✅ All required configuration loaded successfully\n")
    
    return len(issues) == 0


# ==================== HELPER FUNCTIONS ====================

def get_gemini_api_key():
    """Get Gemini API key (for use in other modules)"""
    return GEMINI_API_KEY

def is_gemini_configured():
    """Check if Gemini API is configured"""
    return bool(GEMINI_API_KEY)

# ==================== AUTO-RUN ON IMPORT ====================

if __name__ == "__main__":
    validate_config()
