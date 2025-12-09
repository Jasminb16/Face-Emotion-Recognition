#!/usr/bin/env python3
"""
Simple Data Loader for Emotion Recognition
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_emotion_data(data_path="data/emotions"):
    """Load emotion images from folder structure"""
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    print("📂 Loading emotion dataset...")
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Load training data
    train_path = os.path.join(data_path, 'train')
    for emotion_idx, emotion in enumerate(emotions):
        emotion_path = os.path.join(train_path, emotion)
        if os.path.exists(emotion_path):
            images = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   Training {emotion}: {len(images)} images")
            
            for img_name in images:
                img_path = os.path.join(emotion_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    X_train.append(img)
                    y_train.append(emotion_idx)
    
    # Load test data
    test_path = os.path.join(data_path, 'test')
    for emotion_idx, emotion in enumerate(emotions):
        emotion_path = os.path.join(test_path, emotion)
        if os.path.exists(emotion_path):
            images = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   Testing {emotion}: {len(images)} images")
            
            for img_name in images:
                img_path = os.path.join(emotion_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    X_test.append(img)
                    y_test.append(emotion_idx)
    
    # Convert to numpy arrays
    X_train = np.array(X_train).reshape(-1, 48, 48, 1) / 255.0
    y_train = np.array(y_train)
    X_test = np.array(X_test).reshape(-1, 48, 48, 1) / 255.0
    y_test = np.array(y_test)
    
    print(f"✅ Dataset loaded:")
    print(f"   Training: {X_train.shape[0]} images")
    print(f"   Testing: {X_test.shape[0]} images")
    
    return X_train, y_train, X_test, y_test