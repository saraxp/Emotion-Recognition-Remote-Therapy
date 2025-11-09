"""
Convert FER2013 image folders to CSV format
Run this if you have train/test folders instead of fer2013.csv
"""

import os
import cv2
import pandas as pd
from pathlib import Path
import numpy as np

# Emotion mapping
emotion_dict = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

def images_to_csv(base_path, output_csv='fer2013.csv'):
    """
    Convert FER2013 image folders to CSV format
    
    Expected folder structure:
    FER2013/
        train/
            angry/
            disgust/
            fear/
            happy/
            sad/
            surprise/
            neutral/
        test/
            angry/
            ...
    """
    
    data = []
    
    print("Converting images to CSV format...")
    print(f"Base path: {base_path}")
    
    # Process train and test folders
    for usage_folder in ['train', 'test']:
        usage_path = os.path.join(base_path, usage_folder)
        
        if not os.path.exists(usage_path):
            print(f"Warning: {usage_path} not found, skipping...")
            continue
        
        usage_label = 'Training' if usage_folder == 'train' else 'PublicTest'
        
        print(f"\nProcessing {usage_folder} folder...")
        
        # Process each emotion folder
        for emotion_name in os.listdir(usage_path):
            emotion_path = os.path.join(usage_path, emotion_name)
            
            # Skip if not a directory
            if not os.path.isdir(emotion_path):
                continue
            
            emotion_name_lower = emotion_name.lower()
            
            # Skip if not a valid emotion
            if emotion_name_lower not in emotion_dict:
                print(f"  Skipping unknown emotion: {emotion_name}")
                continue
            
            emotion_label = emotion_dict[emotion_name_lower]
            
            # Process all images in this emotion folder
            image_files = [f for f in os.listdir(emotion_path) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"  Processing {emotion_name}: {len(image_files)} images")
            
            for img_file in image_files:
                img_path = os.path.join(emotion_path, img_file)
                
                try:
                    # Read image
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    # Resize to 48x48 if needed
                    if img.shape != (48, 48):
                        img = cv2.resize(img, (48, 48))
                    
                    # Flatten and convert to space-separated string
                    pixels = ' '.join(map(str, img.flatten()))
                    
                    # Add to data
                    data.append({
                        'emotion': emotion_label,
                        'pixels': pixels,
                        'Usage': usage_label
                    })
                    
                except Exception as e:
                    print(f"    Error processing {img_file}: {e}")
                    continue
    
    # Create DataFrame and save
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        
        print(f"\n✓ Conversion complete!")
        print(f"✓ CSV saved as: {output_csv}")
        print(f"✓ Total images: {len(df)}")
        print(f"\nEmotion distribution:")
        print(df['emotion'].value_counts().sort_index())
        print(f"\nUsage distribution:")
        print(df['Usage'].value_counts())
        
        return output_csv
    else:
        print("\n✗ No images found to convert!")
        return None


def verify_csv(csv_path='fer2013.csv'):
    """Verify the generated CSV is correct"""
    print(f"\nVerifying {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
        
        print(f"✓ CSV loaded successfully")
        print(f"✓ Shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Check required columns
        required_cols = ['emotion', 'pixels', 'Usage']
        if all(col in df.columns for col in required_cols):
            print(f"✓ All required columns present")
        else:
            print(f"✗ Missing columns!")
        
        # Sample a few pixels to verify
        sample_pixels = df['pixels'].iloc[0].split()
        print(f"✓ Sample image has {len(sample_pixels)} pixels (should be 2304 for 48x48)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying CSV: {e}")
        return False


if __name__ == "__main__":
    # Update this path to your FER2013 folder location
    base_path = '/Users/parthsmac/Desktop/minor/FER2013'
    
    print("="*60)
    print("FER2013 Image Folders to CSV Converter")
    print("="*60)
    
    # Check if path exists
    if not os.path.exists(base_path):
        print(f"\n✗ Error: Path not found: {base_path}")
        print("\nPlease update the 'base_path' variable in this script")
        print("to point to your FER2013 folder location.")
        exit(1)
    
    # Convert images to CSV
    output_csv = images_to_csv(base_path, output_csv='fer2013.csv')
    
    if output_csv:
        # Verify the CSV
        verify_csv(output_csv)
        
        print("\n" + "="*60)
        print("Next steps:")
        print("  1. The fer2013.csv file has been created")
        print("  2. Run: python train_emotion_model.py")
        print("="*60)
    else:
        print("\n✗ Conversion failed. Please check your folder structure.")
        print("\nExpected structure:")
        print("FER2013/")
        print("  train/")
        print("    angry/")
        print("    disgust/")
        print("    fear/")
        print("    happy/")
        print("    sad/")
        print("    surprise/")
        print("    neutral/")
        print("  test/")
        print("    angry/")
        print("    ...")