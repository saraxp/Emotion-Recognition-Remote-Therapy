"""
Voice Emotion Analysis Module
This module can be integrated with the main application for voice tone analysis
"""

import numpy as np
import librosa
import pyaudio
import wave
from collections import deque
import threading
import os

class VoiceEmotionAnalyzer:
    def __init__(self, model_path=None):
        """
        Initialize Voice Emotion Analyzer
        If no model provided, uses rule-based analysis
        """
        self.model = None
        if model_path and os.path.exists(model_path):
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)
        
        self.emotion_window = deque(maxlen=5)
        self.is_recording = False
        self.audio_frames = []
        
    def extract_features(self, audio_data, sr=22050):
        """
        Extract audio features for emotion analysis
        """
        try:
            # Convert to float
            audio_data = audio_data.astype(float)
            
            # Extract MFCC (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            # Extract Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            
            # Extract Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            
            # Extract energy and pitch
            rms = librosa.feature.rms(y=audio_data)
            
            # Combine all features
            features = np.concatenate([
                mfcc_mean, mfcc_std,
                chroma_mean, chroma_std,
                [np.mean(spectral_centroid)],
                [np.mean(spectral_rolloff)],
                [np.mean(zero_crossing_rate)],
                [np.mean(rms)]
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def analyze_emotion_rule_based(self, features):
        """
        Rule-based emotion detection using audio features
        This is a simplified version - use ML model for better accuracy
        """
        if features is None or len(features) < 30:
            return "Neutral", 0.5
        
        # Extract key features
        mfcc_mean = np.mean(features[:13])
        energy = features[-1]
        pitch_proxy = features[-4]  # spectral centroid
        zcr = features[-2]  # zero crossing rate
        
        # Simple rules (these are approximations)
        emotion = "Neutral"
        confidence = 0.5
        
        # High energy + high pitch = Angry/Excited
        if energy > 0.05 and pitch_proxy > 2000:
            emotion = "Angry"
            confidence = 0.7
        # Low energy + low pitch = Sad
        elif energy < 0.02 and pitch_proxy < 1500:
            emotion = "Sad"
            confidence = 0.65
        # High pitch + variable energy = Fear/Surprise
        elif pitch_proxy > 2500 and zcr > 0.1:
            emotion = "Fear"
            confidence = 0.6
        # Moderate energy + moderate pitch = Happy
        elif 0.03 < energy < 0.06 and 1800 < pitch_proxy < 2200:
            emotion = "Happy"
            confidence = 0.65
        
        return emotion, confidence
    
    def predict_emotion(self, audio_file):
        """
        Predict emotion from audio file
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(audio_file, duration=3, sr=22050)
            
            # Extract features
            features = self.extract_features(audio_data, sr)
            
            if self.model:
                # Use trained model
                features_reshaped = features.reshape(1, -1)
                predictions = self.model.predict(features_reshaped, verbose=0)[0]
                emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Neutral']
                emotion_idx = np.argmax(predictions)
                emotion = emotion_labels[emotion_idx]
                confidence = predictions[emotion_idx]
            else:
                # Use rule-based analysis
                emotion, confidence = self.analyze_emotion_rule_based(features)
            
            return {
                'emotion': emotion,
                'confidence': float(confidence),
                'features': features.tolist() if features is not None else None
            }
            
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return {'emotion': 'Error', 'confidence': 0.0}
    
    def record_audio(self, duration=3, filename='temp_audio.wav'):
        """
        Record audio from microphone
        """
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 22050
        
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            print(f"Recording for {duration} seconds...")
            frames = []
            
            for i in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            print("Recording finished")
            
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            # Save audio file
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            return filename
            
        except Exception as e:
            print(f"Error recording audio: {e}")
            p.terminate()
            return None
    
    def analyze_realtime_audio(self, duration=3):
        """
        Record and analyze audio in real-time
        """
        audio_file = self.record_audio(duration)
        if audio_file:
            result = self.predict_emotion(audio_file)
            # Clean up temp file
            if os.path.exists(audio_file):
                os.remove(audio_file)
            return result
        return None


# Flask integration example
def add_voice_routes(app, analyzer):
    """
    Add voice emotion analysis routes to Flask app
    """
    from flask import jsonify, request
    
    @app.route('/analyze_voice', methods=['POST'])
    def analyze_voice():
        """Analyze uploaded audio file"""
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        temp_path = 'temp_voice.wav'
        audio_file.save(temp_path)
        
        result = analyzer.predict_emotion(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify(result)
    
    @app.route('/start_voice_recording', methods=['POST'])
    def start_voice_recording():
        """Start recording voice in background"""
        duration = request.json.get('duration', 3)
        
        def record_and_analyze():
            result = analyzer.analyze_realtime_audio(duration)
            # Store result or send to session data
            print(f"Voice emotion detected: {result}")
        
        thread = threading.Thread(target=record_and_analyze)
        thread.start()
        
        return jsonify({'status': 'recording', 'duration': duration})


# Training function for voice emotion model
def train_voice_emotion_model(audio_dataset_path, labels_file):
    """
    Train a voice emotion recognition model
    Requires audio dataset (e.g., RAVDESS, TESS, CREMA-D)
    """
    from tensorflow import keras
    from tensorflow.keras import layers
    import pandas as pd
    
    # Load dataset
    print("Loading audio dataset...")
    df = pd.read_csv(labels_file)
    
    X_features = []
    y_labels = []
    
    analyzer = VoiceEmotionAnalyzer()
    
    for idx, row in df.iterrows():
        audio_path = os.path.join(audio_dataset_path, row['filename'])
        if os.path.exists(audio_path):
            features = analyzer.extract_features(
                librosa.load(audio_path, sr=22050)[0]
            )
            if features is not None:
                X_features.append(features)
                y_labels.append(row['emotion'])
    
    X = np.array(X_features)
    y = pd.get_dummies(y_labels).values
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Build model
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(y.shape[1], activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save model
    model.save('voice_emotion_model.h5')
    print("Model saved as voice_emotion_model.h5")
    
    return model


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = VoiceEmotionAnalyzer()
    
    # Test with microphone
    print("Testing voice emotion analysis...")
    print("Speak for 3 seconds after the beep...")
    
    result = analyzer.analyze_realtime_audio(duration=3)
    
    if result:
        print(f"\nDetected Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2%}")
    else:
        print("Failed to analyze audio")