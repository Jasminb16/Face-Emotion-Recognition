#!/usr/bin/env python3
"""
Improved Training for 70%+ Accuracy
Balanced approach: moderate architecture improvements + proven training techniques
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_emotion_data
from models.cnn_models import create_advanced_cnn


def create_improved_cnn():
    """Improved CNN - not too complex, proven architecture"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, BatchNormalization, Activation, MaxPooling2D,
        GlobalAveragePooling2D, Dense, Dropout
    )
    
    model = Sequential([
        # Block 1: 48x48 -> 24x24
        Conv2D(64, 3, padding='same', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, 3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2),
        Dropout(0.25),
        
        # Block 2: 24x24 -> 12x12
        Conv2D(128, 3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, 3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2),
        Dropout(0.25),
        
        # Block 3: 12x12 -> 6x6
        Conv2D(256, 3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, 3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2),
        Dropout(0.3),
        
        # Block 4: 6x6 -> 3x3
        Conv2D(512, 3, padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(2),
        Dropout(0.4),
        
        # Dense
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ], name='improved_emotion_cnn')
    
    return model


def train_improved_model(epochs=100):
    """Train with balanced settings for reliable 68-72% accuracy"""
    
    print("🎭 IMPROVED EMOTION RECOGNITION TRAINING")
    print("=" * 50)
    print("🎯 Target: 68-72% accuracy (proven approach)")
    print("=" * 50)
    
    # Load data
    X_train, y_train, X_test, y_test = load_emotion_data()
    
    if X_train.shape[0] == 0:
        print("❌ No training data found!")
        return
    
    # Split validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print(f"📊 Dataset split:")
    print(f"   Training: {X_train.shape[0]}")
    print(f"   Validation: {X_val.shape[0]}")
    print(f"   Test: {X_test.shape[0]}")
    
    # Moderate augmentation (proven to work)
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create model
    print("🤖 Creating improved CNN...")
    model = create_improved_cnn()
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"📊 Model parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'saved_models/emotion_model_improved.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    print("🏋️ Training improved model...")
    print("⏰ Estimated time: 45-60 minutes")
    
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=64),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    print("\n📊 Final Test Set Evaluation:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    print(f"\n🎉 Training Complete!")
    print(f"📊 Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.1f}%)")
    print(f"📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    print(f"✅ Model saved: saved_models/emotion_model_improved.h5")
    
    # Compare with original
    print(f"\n📈 Improvement over baseline (62.5%):")
    improvement = (test_accuracy - 0.625) * 100
    print(f"   {improvement:+.1f} percentage points")
    
    if test_accuracy >= 0.68:
        print("✨ Successfully achieved target accuracy!")
    elif test_accuracy >= 0.65:
        print("📈 Good improvement! Getting close.")
    
    return test_accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    
    train_improved_model(epochs=args.epochs)
