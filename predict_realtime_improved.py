#!/usr/bin/env python3
"""
Real-time Emotion Detection using Improved Model
"""

import cv2
import numpy as np
import tensorflow as tf

def predict_realtime_improved():
    """Real-time emotion detection from webcam using improved model"""
    
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Load improved model
    try:
        model = tf.keras.models.load_model('saved_models/emotion_model_improved.h5')
        print("✅ Improved model loaded successfully (68% accuracy)")
    except:
        print("❌ Improved model not found! Using original model...")
        try:
            model = tf.keras.models.load_model('saved_models/emotion_model.h5')
            print("✅ Original model loaded (62% accuracy)")
        except:
            print("❌ No model found! Please train first.")
            return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("🎥 Starting real-time emotion detection...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Extract and preprocess face
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.reshape(1, 48, 48, 1) / 255.0
            
            # Predict emotion
            prediction = model.predict(face, verbose=0)
            emotion_idx = np.argmax(prediction)
            confidence = prediction[0][emotion_idx]
            
            emotion = emotions[emotion_idx]
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Emotion Detection - Improved Model', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    predict_realtime_improved()
