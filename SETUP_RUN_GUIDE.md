# Setup & Run Guide

Complete step-by-step guide to set up and run the Facial Emotion Recognition system.

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Git**: For cloning the repository
- **Webcam**: Required for real-time demo
- **GPU** (optional): For faster training (NVIDIA GPU with CUDA recommended)

## 🔧 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ajiteshreddy7/Face-Emotion_Recognition.git
cd Face-Emotion_Recognition
```

### Step 2: Create Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Manual installation (if needed):**
```bash
pip install tensorflow>=2.10.0
pip install opencv-python>=4.6.0
pip install numpy>=1.21.0
pip install scikit-learn>=1.0.0
```

### Step 4: Prepare Dataset

The project expects FER2013 dataset organized as:

```
data/
└── emotions/
    ├── train/
    │   ├── angry/      (3995 images)
    │   ├── disgust/    (436 images)
    │   ├── fear/       (4097 images)
    │   ├── happy/      (7215 images)
    │   ├── neutral/    (4965 images)
    │   ├── sad/        (4830 images)
    │   └── surprise/   (3171 images)
    └── test/
        ├── angry/      (958 images)
        ├── disgust/    (111 images)
        ├── fear/       (1024 images)
        ├── happy/      (1774 images)
        ├── neutral/    (1233 images)
        ├── sad/        (1247 images)
        └── surprise/   (831 images)
```

**To download FER2013:**
1. Visit [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
2. Download the dataset
3. Extract to `data/emotions/` folder
4. Organize images by emotion into train/test splits

### Step 5: Verify Installation

```bash
python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

---

## 🚀 Running the System

### Option 1: Use Pre-trained Models (Recommended)

The best pre-trained models are already included in `saved_models/`:

**68% Accuracy Model** (Recommended - Better Performance):
```bash
python predict_realtime_improved.py
```

**62% Accuracy Model** (Original - Faster Inference):
```bash
python predict_realtime.py
```

**Controls:**
- Point your webcam at your face
- Emotion labels appear in real-time
- Press **'q'** to quit

### Option 2: Train Your Own Models

#### Train 62% Model (Quick - ~20 minutes)
```bash
python train_simple.py
```

**Output:** Saves to `saved_models/emotion_model_new.h5`

#### Train 68% Model (Better - ~45 minutes)
```bash
python train_improved.py --epochs 60
```

**Output:** Saves to `saved_models/emotion_model_improved.h5`

**Custom epoch count:**
```bash
python train_improved.py --epochs 100  # For more training
python train_improved.py --epochs 40   # For quick training
```

---

## 📊 Training Details

### Simple Model (62%)
```bash
python train_simple.py
```

**What it does:**
- Trains on full training set
- Uses data augmentation
- Early stopping (patience=15)
- No validation split
- Saves best model automatically

**Expected output:**
```
🎭 EMOTION RECOGNITION TRAINING
==================================================
📂 Loading emotion dataset...
   Training angry: 3995 images
   ...
✅ Dataset loaded:
   Training: 28709 images
   Testing: 7178 images
🤖 Creating CNN model...
📊 Model parameters: 685,159
🏋️ Training model...
Epoch 1/50
...
🎉 Training Complete!
📊 Final Accuracy: 0.6195 (62.0%)
✅ Model saved: saved_models/emotion_model_new.h5
```

### Improved Model (68%)
```bash
python train_improved.py --epochs 60
```

**What it does:**
- Trains on 90% training set
- Uses validation split (10%)
- Better data augmentation
- Learning rate scheduling
- Early stopping (patience=20)
- Saves best model automatically

**Expected output:**
```
🎭 IMPROVED EMOTION RECOGNITION TRAINING
==================================================
🎯 Target: 68-72% accuracy (proven approach)
==================================================
📂 Loading emotion dataset...
✅ Dataset loaded:
   Training: 25838 images
   Validation: 2871 images
   Test: 7178 images
🤖 Creating improved CNN...
📊 Model parameters: 2,728,903
🏋️ Training improved model...
⏰ Estimated time: 45-60 minutes
Epoch 1/60
...
🎉 Training Complete!
📊 Validation Accuracy: 0.6789 (67.9%)
📊 Test Accuracy: 0.6789 (67.9%)
✅ Model saved: saved_models/emotion_model_improved.h5
```

---

## 🎯 Expected Results

### 62% Model
- **Training time**: ~20 minutes
- **Accuracy**: 62-63%
- **Parameters**: 685K
- **File size**: 8.3 MB
- **Best for**: Quick inference, resource-constrained systems

### 68% Model
- **Training time**: ~45-50 minutes
- **Accuracy**: 67-68%
- **Parameters**: 2.7M
- **File size**: 10.4 MB
- **Best for**: Production systems, best accuracy

---

## 🔍 Testing the Models

### Quick Test with Saved Models

```bash
# Test improved model (68%)
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('saved_models/emotion_model_improved.h5')
print(f'Model loaded successfully')
model.summary()
"
```

### Real-time Demo

```bash
# Run with improved model
python predict_realtime_improved.py

# Or run with original model
python predict_realtime.py
```

**Demo instructions:**
1. Allow webcam access when prompted
2. Position face in front of camera
3. See emotion prediction in real-time
4. Press 'q' to exit

---

## 🛠️ Troubleshooting

### Issue: "Module not found: tensorflow"
**Solution:**
```bash
pip install --upgrade tensorflow
```

### Issue: "No module named 'cv2'"
**Solution:**
```bash
pip install opencv-python
```

### Issue: "No webcam detected"
**Solution:**
- Check if webcam is plugged in
- Check if other apps are using webcam
- Try restarting the script

### Issue: Low accuracy during training
**Solution:**
- Ensure dataset is properly organized
- Check image quality (should be 48×48 grayscale)
- Try training for more epochs
- Check if data augmentation is too aggressive

### Issue: Out of memory (CUDA)
**Solution:**
- Reduce batch size in script
- Use CPU instead (slower but works)
- Close other GPU-intensive apps

### Issue: Training takes too long
**Solution:**
- Use smaller model: `python train_simple.py`
- Reduce epochs: `python train_improved.py --epochs 30`
- Use GPU acceleration if available

---

## 📊 Monitoring Training

### Watch Training Progress

**Windows (PowerShell):**
```powershell
# Real-time monitoring (if using output file)
Get-Content train_output.txt -Wait
```

**macOS/Linux:**
```bash
# Real-time monitoring (if using output file)
tail -f train_output.txt
```

### Check Model Files

```bash
# List saved models
Get-ChildItem saved_models/ -Filter *.h5

# Check model size
$model = Get-Item saved_models/emotion_model_improved.h5
Write-Host "Size: $($model.Length / 1MB) MB"
```

---

## 🔄 Using Different Models

### Switch Between Models

**Use 68% model (Better accuracy):**
```bash
# Edit predict_realtime_improved.py or
python predict_realtime_improved.py
```

**Use 62% model (Faster inference):**
```bash
# Edit predict_realtime.py or
python predict_realtime.py
```

**Use custom model:**
```python
import tensorflow as tf
model = tf.keras.models.load_model('saved_models/emotion_model_improved.h5')
# Your code here
```

---

## 🚀 Performance Optimization

### For Faster Inference

1. **Use 62% model instead of 68%:**
   ```bash
   python predict_realtime.py
   ```

2. **Convert to TensorFlow Lite** (optional):
   ```python
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   ```

3. **Use quantization** (reduces model size):
   - See TensorFlow docs for model quantization

### For Better Accuracy

1. **Use 68% model:**
   ```bash
   python predict_realtime_improved.py
   ```

2. **Train longer:**
   ```bash
   python train_improved.py --epochs 100
   ```

3. **Collect more data:**
   - Add more emotion images to dataset

---

## 📝 Project Structure Summary

```
Face-Emotion_Recognition/
├── data/
│   └── emotions/               ← Your dataset
│       ├── train/
│       └── test/
├── models/
│   └── cnn_models.py          ← CNN architectures
├── utils/
│   └── data_loader.py         ← Data loading
├── saved_models/              ← Trained models
│   ├── emotion_model.h5       ← 62% model (backup)
│   └── emotion_model_improved.h5 ← 68% model ⭐
├── train_simple.py            ← Train 62% model
├── train_improved.py          ← Train 68% model
├── predict_realtime.py        ← Demo (62%)
├── predict_realtime_improved.py ← Demo (68%) ⭐
├── requirements.txt
├── README.md                  ← Project overview
└── SETUP_RUN_GUIDE.md        ← This file
```

---

## ✅ Verification Checklist

After setup, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All requirements installed
- [ ] Dataset in `data/emotions/` folder
- [ ] Can import tensorflow and cv2
- [ ] Models exist in `saved_models/`
- [ ] Webcam working

---

## 🎓 Next Steps

1. **Run the demo:** `python predict_realtime_improved.py`
2. **Explore the code:** Check `models/cnn_models.py`
3. **Train your own:** `python train_improved.py`
4. **Read the full README:** See `README.md` for architecture details
5. **Customize:** Modify training parameters as needed

---

## 📞 Support

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Verify dataset is correctly organized
3. Ensure Python version is 3.8+
4. Try reinstalling dependencies
5. Check GitHub issues

---

**Quick Start Command:**
```bash
# Copy & paste to run immediately:
python predict_realtime_improved.py
```

---

