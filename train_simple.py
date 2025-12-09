#!/usr/bin/env python3
"""
Simple Training Script for Emotion Recognition
Original version that achieved 62.5% accuracy
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.data_loader import load_emotion_data
from models.cnn_models import create_simple_cnn, create_advanced_cnn

def train_emotion_model():
    """Train the emotion recognition model"""
    
    print("🎭 EMOTION RECOGNITION TRAINING")
    print("=" * 50)
    
    # Load data
    X_train, y_train, X_test, y_test = load_emotion_data()
    
    if X_train.shape[0] == 0:
        print("❌ No training data found!")
        print("Please add images to data/emotions/train/[emotion_name]/")
        return
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    
    # Create model
    print("🤖 Creating CNN model...")
    model = create_advanced_cnn()
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"📊 Model parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'saved_models/emotion_model_new.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train model
    print("🏋️ Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n🎉 Training Complete!")
    print(f"📊 Final Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    print(f"✅ Model saved: saved_models/emotion_model_new.h5")
    
    return test_accuracy

if __name__ == "__main__":
    train_emotion_model()
