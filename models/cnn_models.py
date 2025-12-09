#!/usr/bin/env python3
"""
Simple CNN Models for Emotion Recognition
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    Input,
    GlobalAveragePooling2D,
    Resizing,
    Lambda,
)

def create_simple_cnn():
    """Create a simple CNN model"""
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotions
    ])
    
    return model

def create_advanced_cnn():
    """Create an advanced CNN model"""
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotions
    ])
    
    return model


def create_mobilenet_v2(image_size: int = 96, dropout: float = 0.3):
    """Create a transfer-learning model based on MobileNetV2.

    - Accepts 48x48 grayscale inputs and internally upsamples to RGB.
    - Starts with the base model frozen; unfreeze selected layers in training.
    """

    # Input: keep compatibility with existing grayscale pipeline
    inputs = Input(shape=(48, 48, 1))

    # Resize to a size MobileNetV2 is happy with and expand to 3 channels
    x = Resizing(image_size, image_size, name="resize_to_mobilenet")(inputs)
    x = Lambda(lambda t: tf.repeat(t, repeats=3, axis=-1), name="grayscale_to_rgb")(x)
    x = preprocess_input(x)

    base_model = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
    )
    base_model.trainable = False

    x = base_model(x)
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Dropout(dropout, name="post_pool_dropout")(x)
    x = Dense(256, activation="relu", name="dense_256")(x)
    x = BatchNormalization(name="bn_256")(x)
    x = Dropout(dropout, name="head_dropout")(x)
    outputs = Dense(7, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="mobilenetv2_emotion")

    return model, base_model