import cv2
import numpy as np
import keras
from collections import deque
import datetime
import json
import os

# Load the trained model
model = keras.models.load_model('emotion_model_best.h5')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load face cascade for face detection

try:
    # Try using cv2.data first
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # type: ignore
except:
    # Fallback to manual path
    cascade_path = os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)

# Store emotion history for smoothing
emotion_window = deque(maxlen=10)

# Store session data
session_data = {
    'start_time': datetime.datetime.now().isoformat(),
    'emotions_detected': [],
    'emotion_timeline': []
}

def preprocess_face(face_img):
    """Preprocess face image for model prediction"""
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    return face_img

def get_smoothed_emotion(predictions):
    """Smooth emotion predictions using a sliding window"""
    emotion_window.append(predictions)
    avg_predictions = np.mean(emotion_window, axis=0)
    return avg_predictions

def draw_emotion_bar(frame, emotion_probs, x, y):
    """Draw emotion probability bars on frame"""
    bar_height = 20
    bar_max_width = 150
    start_y = y
    
    for i, (label, prob) in enumerate(zip(emotion_labels, emotion_probs)):
        bar_width = int(prob * bar_max_width)
        color = (0, 255, 0) if i == np.argmax(emotion_probs) else (100, 100, 100)
        
        # Draw bar
        cv2.rectangle(frame, (x, start_y), (x + bar_width, start_y + bar_height - 5), color, -1)
        
        # Draw label and percentage
        text = f"{label}: {prob*100:.1f}%"
        cv2.putText(frame, text, (x + bar_max_width + 10, start_y + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        start_y += bar_height + 5

def save_session_data(filename='session_data.json'):
    """Save session data to JSON file"""
    session_data['end_time'] = datetime.datetime.now().isoformat()
    with open(filename, 'w') as f:
        json.dump(session_data, f, indent=4)
    print(f"Session data saved to {filename}")

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting emotion detection. Press 'q' to quit, 's' to save session data.")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_count += 1
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess face
            processed_face = preprocess_face(face_roi)
            
            # Predict emotion
            predictions = model.predict(processed_face, verbose=0)[0] # type: ignore
            
            # Smooth predictions
            smoothed_predictions = get_smoothed_emotion(predictions)
            
            # Get dominant emotion
            emotion_idx = np.argmax(smoothed_predictions)
            emotion = emotion_labels[emotion_idx]
            confidence = smoothed_predictions[emotion_idx]
            
            # Record emotion every 30 frames (about 1 second at 30fps)
            if frame_count % 30 == 0:
                timestamp = datetime.datetime.now().isoformat()
                emotion_data = {
                    'timestamp': timestamp,
                    'emotion': emotion,
                    'confidence': float(confidence),
                    'all_probabilities': {label: float(prob) for label, prob in zip(emotion_labels, smoothed_predictions)}
                }
                session_data['emotions_detected'].append(emotion_data)
                
                # Update emotion timeline
                if emotion not in [e['emotion'] for e in session_data['emotion_timeline']]:
                    session_data['emotion_timeline'].append({'emotion': emotion, 'first_detected': timestamp})
            
            # Draw rectangle around face
            color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label
            label = f"{emotion} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw emotion probability bars
            if x + w + 200 < frame.shape[1]:  # Check if there's space on the right
                draw_emotion_bar(frame, smoothed_predictions, x + w + 10, y)
        
        # Display info
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('Emotion Recognition for Remote Therapy', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_session_data()
    
    # Cleanup
    save_session_data()  # Auto-save on exit
    cap.release()
    cv2.destroyAllWindows()
    
    # Print session summary
    print("\n=== Session Summary ===")
    print(f"Total emotions detected: {len(session_data['emotions_detected'])}")
    if session_data['emotions_detected']:
        emotion_counts = {}
        for data in session_data['emotions_detected']:
            emotion = data['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("\nEmotion distribution:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(session_data['emotions_detected'])) * 100
            print(f"  {emotion}: {count} times ({percentage:.1f}%)")

if __name__ == "__main__":
    main()