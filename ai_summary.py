"""
AI Summary Generator using Google Gemini API
Generates therapy session summaries from emotion data
"""

import json
from datetime import datetime
from collections import Counter

# Type checking fix for Pylance
try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None  # type: ignore

# Import configuration from existing config.py
from config import GEMINI_API_KEY, GEMINI_MODEL

def initialize_gemini():
    """Initialize Gemini API with API key"""
    if not GEMINI_API_KEY or genai is None:
        print("⚠️  GEMINI_API_KEY not found in environment variables")
        return False
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)  # type: ignore
        return True
    except Exception as e:
        print(f"❌ Error initializing Gemini: {e}")
        return False

def check_gemini_available():
    """Check if Gemini API is configured and available"""
    if not GEMINI_API_KEY or genai is None:
        return False
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)  # type: ignore
        # Try to initialize a model to verify API key works
        model = genai.GenerativeModel(GEMINI_MODEL)  # type: ignore
        return True
    except Exception as e:
        print(f"❌ Gemini API not available: {e}")
        return False

def generate_summary_with_gemini(session_data, patient_name):
    """
    Generate AI summary using Google Gemini API
    Falls back to rule-based summary if Gemini is not available
    """
    
    if not check_gemini_available():
        print("⚠️  Gemini not available, using rule-based summary")
        return generate_rule_based_summary(session_data, patient_name)
    
    try:
        # Prepare emotion data
        emotions = session_data.get('emotions', [])
        
        # Calculate duration first (we need this regardless)
        start_time = datetime.fromisoformat(session_data['start_time'])
        end_time = datetime.fromisoformat(session_data.get('end_time', datetime.now().isoformat()))
        duration_minutes = int((end_time - start_time).total_seconds() / 60)
        duration_seconds = int((end_time - start_time).total_seconds())
        
        # ✅ FIX: Handle sessions with no emotion data
        if not emotions or len(emotions) == 0:
            prompt = f"""You are a professional therapist assistant. Generate a brief, professional summary for a therapy session where no facial emotion data was captured.

Session Details:
- Patient: {patient_name}
- Duration: {duration_minutes} minutes {duration_seconds % 60} seconds
- Total emotion detections: 0
- Note: No facial expressions were detected during this session

Generate a concise 2-3 sentence professional summary that:
1. Acknowledges the session took place but emotion tracking was unavailable
2. Suggests possible reasons (camera issues, patient not visible, technical difficulties)

Do NOT include:
- "Session Summary for [name]:" header
- "Clinical Recommendation:" section

Keep it professional and constructive. Write in paragraph form only."""
        else:
            # Calculate statistics
            emotion_counts = Counter([e['emotion'] for e in emotions])
            total_emotions = len(emotions)
            
            # Get dominant emotions
            top_emotions = emotion_counts.most_common(3)
            
            # Calculate average confidence
            avg_confidence = sum([e['confidence'] for e in emotions]) / total_emotions
            
            # Create prompt for Gemini
            prompt = f"""
                You are a professional therapist assistant. Generate a brief, professional summary of a therapy session.

                Session Details:
                - Patient: {patient_name}
                - Duration: {duration_minutes} minutes {duration_seconds % 60} seconds
                - Total emotion detections: {total_emotions}
                - Average confidence: {avg_confidence:.1%}

                Emotion Distribution:
                {chr(10).join([f"- {emotion}: {count} times ({(count/total_emotions)*100:.1f}%)" for emotion, count in top_emotions])}

                Generate a concise paragraph (3-4 sentences) focusing on:
                1. Overall emotional state during the session
                2. Most prevalent emotions and what they might indicate
                3. Any notable patterns or observations
                Write only the summary paragraph in a professional, empathetic tone. Do not include an additional header or clinical recommendations.

            """

        # Initialize Gemini model
        model = genai.GenerativeModel(GEMINI_MODEL)  # type: ignore
        
        # Generate content
        response = model.generate_content(  # type: ignore
            prompt,
            generation_config=genai.types.GenerationConfig(  # type: ignore
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=500,
            )
        )
        
        if response and response.text:
            summary = response.text.strip()
            if summary:
                print("✅ AI summary generated successfully with Gemini")
                return summary
        
        # Fallback if API call fails
        print("⚠️  Gemini response was empty, using rule-based summary")
        return generate_rule_based_summary(session_data, patient_name)
        
    except Exception as e:
        print(f"❌ Error generating Gemini summary: {e}")
        import traceback
        traceback.print_exc()
        return generate_rule_based_summary(session_data, patient_name)

def generate_rule_based_summary(session_data, patient_name):
    """
    Generate rule-based summary without AI
    Used as fallback when Gemini is not available
    """
    
    emotions = session_data.get('emotions', [])
    
    # Calculate duration
    start_time = datetime.fromisoformat(session_data['start_time'])
    end_time = datetime.fromisoformat(session_data.get('end_time', datetime.now().isoformat()))
    duration_minutes = int((end_time - start_time).total_seconds() / 60)
    duration_seconds = int((end_time - start_time).total_seconds())
    
    # ✅ FIX: Handle no emotion data case
    if not emotions or len(emotions) == 0:
        return f"""A {duration_minutes}-minute {duration_seconds % 60}-second session was conducted with {patient_name}, however no facial emotion data was captured during this period. This may indicate technical difficulties with the camera system, the patient not being visible to the camera, or connectivity issues during the session."""
    
    # Calculate statistics
    emotion_counts = Counter([e['emotion'] for e in emotions])
    total_emotions = len(emotions)
    
    # Get dominant emotion
    dominant_emotion, dominant_count = emotion_counts.most_common(1)[0]
    dominant_percentage = (dominant_count / total_emotions) * 100
    
    # Build summary based on dominant emotion
    emotion_interpretations = {
        'Happy': {
            'description': 'positive and upbeat mood',
            'recommendation': 'Continue to reinforce positive coping strategies and maintain current therapeutic approach.'
        },
        'Sad': {
            'description': 'sadness and possible low mood',
            'recommendation': 'Focus on exploring underlying causes and developing emotional regulation techniques.'
        },
        'Angry': {
            'description': 'frustration or anger',
            'recommendation': 'Address anger management strategies and identify triggers for emotional responses.'
        },
        'Fear': {
            'description': 'anxiety or fear',
            'recommendation': 'Implement relaxation techniques and gradually address anxiety-inducing topics.'
        },
        'Surprise': {
            'description': 'unexpected reactions',
            'recommendation': 'Explore the sources of surprise and their emotional significance.'
        },
        'Neutral': {
            'description': 'emotionally stable and composed demeanor',
            'recommendation': 'Continue current approach while monitoring for any subtle emotional changes.'
        },
        'Disgust': {
            'description': 'aversion or discomfort',
            'recommendation': 'Identify sources of discomfort and work on acceptance-based strategies.'
        }
    }
    
    interp = emotion_interpretations.get(dominant_emotion, {
        'description': 'varied emotional responses',
        'recommendation': 'Continue monitoring emotional patterns and adjust therapeutic approach as needed.'
    })
    
    # Get other significant emotions
    other_emotions = [e for e, _ in emotion_counts.most_common(3)[1:]]
    other_text = f" Additional emotions observed include {', '.join(other_emotions)}." if other_emotions else ""
    
    summary = f"""During this {duration_minutes}-minute {duration_seconds % 60}-second session, {patient_name} predominantly displayed {dominant_emotion.lower()} emotions ({dominant_percentage:.1f}% of observed expressions), indicating a {interp['description']}.{other_text} Based on {total_emotions} emotion detections with consistent monitoring throughout the session, the patient's emotional state appears to be characterized by {dominant_emotion.lower()} affect."""
    
    return summary

def generate_quick_summary(emotion_counts, duration_minutes):
    """Generate a quick one-line summary"""
    if not emotion_counts:
        return "No emotions detected"
    
    dominant_emotion, dominant_count = emotion_counts.most_common(1)[0]
    total = sum(emotion_counts.values())
    percentage = (dominant_count / total) * 100
    
    return f"{duration_minutes}min session - Predominantly {dominant_emotion} ({percentage:.0f}%)"

# ==================== GEMINI SETUP INSTRUCTIONS ====================

def print_gemini_setup_instructions():
    """Print instructions for setting up Gemini API"""
    instructions = """
╔═══════════════════════════════════════════════════════════════════╗
║           GOOGLE GEMINI API SETUP INSTRUCTIONS (FREE)             ║
╚═══════════════════════════════════════════════════════════════════╝

Gemini offers FREE API access with generous limits!

STEP 1: Get Your API Key
------------------------
1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy your API key

STEP 2: Set Environment Variable
------------------------
Windows (Command Prompt):
  set GEMINI_API_KEY=your_api_key_here

Windows (PowerShell):
  $env:GEMINI_API_KEY="your_api_key_here"

Mac/Linux:
  export GEMINI_API_KEY=your_api_key_here

For permanent setup, add to your system environment variables.

STEP 3: Install Required Package
------------------------
  pip install google-generativeai

STEP 4: Test It
------------------------
In Python:
  python ai_summary.py

GEMINI FREE TIER LIMITS:
------------------------
- 15 requests per minute
- 1 million tokens per minute
- 1500 requests per day
- Completely FREE for development and production use!

TROUBLESHOOTING:
------------------------
If API key is not recognized:
  - Check that environment variable is set correctly
  - Restart your terminal/IDE after setting the variable
  - Verify API key at https://makersuite.google.com/app/apikey

If rate limit exceeded:
  - Wait 1 minute and try again
  - The app will automatically fall back to rule-based summaries

╔═══════════════════════════════════════════════════════════════════╗
║  After setup, your summaries will be AI-generated automatically   ║
║  If Gemini is not available, rule-based summaries will be used    ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(instructions)

# ==================== TESTING ====================

if __name__ == "__main__":
    print("="*70)
    print("AI SUMMARY GENERATOR - TEST (GEMINI)")
    print("="*70)
    
    # Check if Gemini is available
    if check_gemini_available():
        print("\n✅ Gemini API is configured and ready!")
        
        # Test with emotions
        test_session_with_emotions = {
            'session_id': 'test_123',
            'start_time': '2024-01-15T10:00:00',
            'end_time': '2024-01-15T10:45:00',
            'emotions': [
                {'emotion': 'Happy', 'confidence': 0.85, 'timestamp': '2024-01-15T10:05:00'},
                {'emotion': 'Happy', 'confidence': 0.82, 'timestamp': '2024-01-15T10:10:00'},
                {'emotion': 'Neutral', 'confidence': 0.75, 'timestamp': '2024-01-15T10:15:00'},
                {'emotion': 'Sad', 'confidence': 0.70, 'timestamp': '2024-01-15T10:20:00'},
                {'emotion': 'Happy', 'confidence': 0.88, 'timestamp': '2024-01-15T10:25:00'},
            ]
        }
        
        print("\n" + "="*70)
        print("TEST 1: Session WITH emotions")
        print("="*70)
        summary = generate_summary_with_gemini(test_session_with_emotions, "John Doe")
        print(summary)
        
        # Test without emotions (like your case)
        test_session_no_emotions = {
            'session_id': 'test_456',
            'start_time': '2025-11-07T10:00:00',
            'end_time': '2025-11-07T10:00:09',
            'emotions': []
        }
        
        print("\n" + "="*70)
        print("TEST 2: Session WITHOUT emotions (your scenario)")
        print("="*70)
        summary = generate_summary_with_gemini(test_session_no_emotions, "Vaibhavi")
        print(summary)
        print("="*70)
        
    else:
        print("\n❌ Gemini API key not found or invalid")
        print("\nTo enable AI summaries, follow these instructions:\n")
        print_gemini_setup_instructions()
