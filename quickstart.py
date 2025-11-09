"""
Quick Start Script for Emotion Recognition Project
This script helps set up and run the project with minimal manual intervention
"""

import os
import sys
import subprocess
import urllib.request
from pathlib import Path

class ProjectSetup:
    def __init__(self):
        self.project_root = Path.cwd()
        self.required_dirs = ['templates', 'session_data', 'static', 'models']
        self.required_files = {
            'requirements.txt': True,
            'fer2013.csv': False,
            'emotion_model_best.h5': False
        }
        
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "="*60)
        print(f"  {text}")
        print("="*60 + "\n")
    
    def create_directories(self):
        """Create required project directories"""
        self.print_header("Creating Project Directories")
        for dir_name in self.required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
                print(f"✓ Created: {dir_name}/")
            else:
                print(f"✓ Exists: {dir_name}/")
    
    def check_files(self):
        """Check if required files exist"""
        self.print_header("Checking Required Files")
        missing_files = []
        
        for file_name, required in self.required_files.items():
            file_path = self.project_root / file_name
            if file_path.exists():
                print(f"✓ Found: {file_name}")
            else:
                if required:
                    print(f"✗ Missing (Required): {file_name}")
                    missing_files.append(file_name)
                else:
                    print(f"⚠ Missing (Optional): {file_name}")
        
        return missing_files
    
    def install_dependencies(self):
        """Install Python dependencies"""
        self.print_header("Installing Dependencies")
        
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            print("✗ requirements.txt not found!")
            print("Please create requirements.txt first")
            return False
        
        try:
            print("Installing packages... This may take several minutes.")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ])
            print("✓ All dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing dependencies: {e}")
            return False
    
    def check_webcam(self):
        """Test if webcam is accessible"""
        self.print_header("Testing Webcam")
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✓ Webcam is accessible")
                cap.release()
                return True
            else:
                print("✗ Webcam not accessible")
                return False
        except Exception as e:
            print(f"✗ Error testing webcam: {e}")
            return False
    
    def download_haarcascade(self):
        """Download Haar Cascade file if not present"""
        self.print_header("Checking Face Detection Model")
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml' # type: ignore
            if os.path.exists(cascade_path):
                print("✓ Haar Cascade file found")
                return True
            else:
                print("✗ Haar Cascade not found in OpenCV installation")
                return False
        except Exception as e:
            print(f"⚠ Could not verify Haar Cascade: {e}")
            return True  # Don't block setup
    
    def verify_model(self):
        """Check if emotion recognition model exists"""
        self.print_header("Checking Emotion Recognition Model")
        
        model_path = self.project_root / 'emotion_model_best.h5'
        
        if model_path.exists():
            print("✓ Emotion model found: emotion_model_best.h5")
            return True
        else:
            print("✗ Emotion model not found")
            print("\nYou need to train the model first:")
            print("  1. Download FER2013 dataset from Kaggle")
            print("  2. Place fer2013.csv in project root")
            print("  3. Run: python train_emotion_model.py")
            return False
    
    def create_config_file(self):
        """Create a configuration file"""
        config_content = """# Emotion Recognition Configuration

# Flask Settings
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# Model Settings
MODEL_PATH = 'emotion_model_best.h5'
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
"""
        config_path = self.project_root / 'config.py'
        if not config_path.exists():
            with open(config_path, 'w') as f:
                f.write(config_content)
            print("✓ Created config.py")
        else:
            print("✓ config.py already exists")
    
    def run_full_setup(self):
        """Run complete project setup"""
        print("\n" + "="*60)
        print("  EMOTION RECOGNITION PROJECT - QUICK SETUP")
        print("="*60)
        
        # Step 1: Create directories
        self.create_directories()
        
        # Step 2: Check files
        missing = self.check_files()
        if missing:
            print(f"\n⚠ Warning: Missing required files: {', '.join(missing)}")
        
        # Step 3: Install dependencies
        install_choice = input("\nInstall Python dependencies? (y/n): ").lower()
        if install_choice == 'y':
            self.install_dependencies()
        
        # Step 4: Check webcam
        self.check_webcam()
        
        # Step 5: Check Haar Cascade
        self.download_haarcascade()
        
        # Step 6: Verify model
        has_model = self.verify_model()
        
        # Step 7: Create config
        self.create_config_file()
        
        # Final summary
        self.print_header("Setup Summary")
        
        if has_model:
            print("✓ Setup complete! You can now run the application.")
            print("\nNext steps:")
            print("  1. Start the server: python app.py")
            print("  2. Open browser: http://localhost:5000")
            print("  3. View dashboard: http://localhost:5000/dashboard")
        else:
            print("⚠ Setup incomplete - Model not found")
            print("\nNext steps:")
            print("  1. Download FER2013 from Kaggle: https://www.kaggle.com/datasets/msambare/fer2013")
            print("  2. Extract fer2013.csv to project root")
            print("  3. Run training: python train_emotion_model.py")
            print("  4. After training, run: python quickstart.py (to verify)")
            print("  5. Start server: python app.py")
        
        print("\n" + "="*60 + "\n")


def show_menu():
    """Show interactive menu"""
    print("\n" + "="*60)
    print("  EMOTION RECOGNITION - QUICK START MENU")
    print("="*60)
    print("\n1. Run Full Setup")
    print("2. Install Dependencies Only")
    print("3. Test Webcam")
    print("4. Check Model Status")
    print("5. Start Flask Application")
    print("6. Train Model (if dataset available)")
    print("7. Run Standalone Detection (no web)")
    print("8. View System Requirements")
    print("9. Exit")
    print("\n" + "="*60)


def install_dependencies_only():
    """Install only the Python packages"""
    try:
        print("\nInstalling dependencies...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✓ Installation complete!")
    except Exception as e:
        print(f"✗ Installation failed: {e}")


def test_webcam_only():
    """Quick webcam test"""
    try:
        import cv2
        print("\nTesting webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Cannot access webcam")
            return
        
        print("✓ Webcam accessible!")
        print("Opening camera preview for 5 seconds...")
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Webcam Test - Press Q to quit', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Webcam test complete!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure OpenCV is installed: pip install opencv-python")


def check_model_status():
    """Check emotion model status"""
    print("\nChecking model status...")
    
    model_files = ['emotion_model_best.h5', 'emotion_model_final.h5']
    found = False
    
    for model_file in model_files:
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"✓ Found: {model_file} ({file_size:.2f} MB)")
            found = True
            
            # Try to load model
            try:
                import keras
                model = keras.models.load_model(model_file)
                print(f"  Model loaded successfully!")
                print(f"  Input shape: {model.input_shape}") # type: ignore
                print(f"  Output shape: {model.output_shape}") # type: ignore
            except Exception as e:
                print(f"  ⚠ Warning: Could not load model: {e}")
    
    if not found:
        print("✗ No model files found")
        print("\nTo train a model:")
        print("  1. Download FER2013 dataset")
        print("  2. Run: python train_emotion_model.py")
    
    # Check dataset
    print("\nChecking dataset...")
    if os.path.exists('fer2013.csv'):
        file_size = os.path.getsize('fer2013.csv') / (1024 * 1024)  # MB
        print(f"✓ Found: fer2013.csv ({file_size:.2f} MB)")
    else:
        print("✗ fer2013.csv not found")


def start_flask_app():
    """Start the Flask application"""
    if not os.path.exists('app.py'):
        print("✗ app.py not found in current directory")
        return
    
    if not os.path.exists('emotion_model_best.h5'):
        print("⚠ Warning: emotion_model_best.h5 not found")
        proceed = input("Continue anyway? (y/n): ").lower()
        if proceed != 'y':
            return
    
    print("\nStarting Flask application...")
    print("Access the application at:")
    print("  - Main page: http://localhost:5000")
    print("  - Dashboard: http://localhost:5000/dashboard")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.call([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n\nServer stopped.")


def train_model():
    """Train the emotion recognition model"""
    if not os.path.exists('fer2013.csv'):
        print("✗ fer2013.csv not found!")
        print("\nPlease download the FER2013 dataset:")
        print("  1. Go to: https://www.kaggle.com/datasets/msambare/fer2013")
        print("  2. Download and extract fer2013.csv")
        print("  3. Place it in the project root directory")
        return
    
    if not os.path.exists('train_emotion_model.py'):
        print("✗ train_emotion_model.py not found!")
        return
    
    print("\nStarting model training...")
    print("This will take 2-4 hours on CPU (30-60 min on GPU)")
    proceed = input("Continue? (y/n): ").lower()
    
    if proceed == 'y':
        try:
            subprocess.call([sys.executable, 'train_emotion_model.py'])
        except KeyboardInterrupt:
            print("\n\nTraining interrupted.")


def run_standalone_detection():
    """Run standalone emotion detection without web interface"""
    if not os.path.exists('realtime_emotion_detection.py'):
        print("✗ realtime_emotion_detection.py not found!")
        return
    
    if not os.path.exists('emotion_model_best.h5'):
        print("✗ emotion_model_best.h5 not found!")
        print("Please train the model first.")
        return
    
    print("\nStarting standalone emotion detection...")
    print("Press 'q' to quit, 's' to save session")
    
    try:
        subprocess.call([sys.executable, 'realtime_emotion_detection.py'])
    except KeyboardInterrupt:
        print("\n\nDetection stopped.")


def show_requirements():
    """Display system requirements"""
    print("\n" + "="*60)
    print("  SYSTEM REQUIREMENTS")
    print("="*60)
    print("\nSoftware Requirements:")
    print("  - Python 3.8 or higher")
    print("  - pip (Python package manager)")
    print("  - Webcam/Camera")
    print("  - 4GB RAM minimum (8GB recommended)")
    print("  - 2GB free disk space")
    
    print("\nPython Packages:")
    print("  - TensorFlow 2.13.0")
    print("  - OpenCV 4.8.0")
    print("  - NumPy 1.24.3")
    print("  - Pandas 2.0.3")
    print("  - Flask 2.3.2")
    print("  - scikit-learn 1.3.0")
    print("  - matplotlib 3.7.2")
    
    print("\nOptional (for voice analysis):")
    print("  - librosa")
    print("  - pyaudio")
    print("  - SpeechRecognition")
    
    print("\nDataset:")
    print("  - FER2013 (35,887 grayscale images, ~300MB)")
    print("  - Download from: kaggle.com/datasets/msambare/fer2013")
    
    print("\nCloud Storage Options (Free Tier):")
    print("  - Firebase: 1GB storage")
    print("  - MongoDB Atlas: 512MB storage")
    print("  - AWS S3: 5GB storage (12 months free)")
    
    print("\nDeployment Options (Free):")
    print("  - PythonAnywhere: 512MB, no credit card")
    print("  - Heroku: 550 dyno hours/month")
    print("  - Railway.app: $5 credit/month")
    print("  - Render.com: Free web services")
    
    print("\n" + "="*60)


def check_python_version():
    """Check if Python version is adequate"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"⚠ Warning: Python {version.major}.{version.minor} detected")
        print("  Python 3.8 or higher is recommended")
        return False
    else:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True


def main():
    """Main function to run the quick start script"""
    
    # Check Python version first
    print("\nChecking Python version...")
    check_python_version()
    
    while True:
        show_menu()
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            setup = ProjectSetup()
            setup.run_full_setup()
        
        elif choice == '2':
            install_dependencies_only()
        
        elif choice == '3':
            test_webcam_only()
        
        elif choice == '4':
            check_model_status()
        
        elif choice == '5':
            start_flask_app()
        
        elif choice == '6':
            train_model()
        
        elif choice == '7':
            run_standalone_detection()
        
        elif choice == '8':
            show_requirements()
        
        elif choice == '9':
            print("\nExiting... Good luck with your project! 🚀\n")
            break
        
        else:
            print("\n✗ Invalid choice. Please enter 1-9.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting...\n")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        print("Please report this issue if it persists.\n")