# Train Kitchen Model on Google Colab (FREE)

Your local system doesn't have enough RAM to train this model. Use Google Colab instead - it's free and has better hardware!

## 🚀 Quick Start (5 minutes)

### Step 1: Upload Dataset to Google Drive

1. Zip your dataset:
```bash
cd /home/athul/sakshi/normal-sakshi
zip -r kitchen_unified_dataset.zip kitchen_unified_dataset/
```

2. Upload `kitchen_unified_dataset.zip` to your Google Drive

### Step 2: Open Google Colab

1. Go to: https://colab.research.google.com/
2. Create new notebook
3. Enable GPU: `Runtime` → `Change runtime type` → `T4 GPU` → `Save`

### Step 3: Run Training Code

Copy-paste this into Colab cells and run:

```python
# Cell 1: Setup
!pip install ultralytics -q

# Cell 2: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Extract dataset
import zipfile
import os

# Update this path to where you uploaded the zip
zip_path = '/content/drive/MyDrive/kitchen_unified_dataset.zip'

# Extract
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('/content/')

print("✅ Dataset extracted!")
print(f"Images: {len(os.listdir('/content/kitchen_unified_dataset/images/train'))}")
print(f"Labels: {len(os.listdir('/content/kitchen_unified_dataset/labels/train'))}")

# Cell 4: Train model
from ultralytics import YOLO
import torch

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Load model
model = YOLO('yolo11n-seg.pt')

# Train (16GB RAM on Colab, can use larger batch!)
results = model.train(
    data='/content/kitchen_unified_dataset/data.yaml',
    epochs=150,
    imgsz=640,
    batch=16,  # Colab has enough RAM!
    device=0,
    workers=2,
    
    project='kitchen_compliance_model',
    name='yolo11n_unified',
    exist_ok=True,
    
    patience=30,
    save=True,
    save_period=10,
    cache=False,
    
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    
    # Optimizer
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    
    # Loss weights
    box=7.5,
    cls=0.5,
    dfl=1.5,
    
    # Training settings
    cos_lr=True,
    close_mosaic=15,
    amp=True,
    verbose=True,
)

print("✅ Training complete!")

# Cell 5: Download trained model
from google.colab import files

# Zip the results
!zip -r trained_model.zip kitchen_compliance_model/yolo11n_unified/weights/

# Download
files.download('trained_model.zip')

print("✅ Model downloaded! Extract and use best.pt")
```

### Step 4: Download and Use Model

After training (4-6 hours):
1. Extract `trained_model.zip`
2. Copy `best.pt` to your local machine: `/home/athul/sakshi/normal-sakshi/kitchen_compliance_model/yolo11n_unified/weights/`
3. Run `python3 test_kitchen_model.py`

## 📊 Colab Advantages

- ✅ **FREE** T4 GPU (16GB VRAM)
- ✅ **12-16GB RAM** (vs your 8GB or less)
- ✅ **Faster training** (4-6 hours vs 8-12 hours)
- ✅ **Better batch size** (16 vs 2)
- ✅ **Full augmentation** enabled
- ✅ **No system crashes**

## ⚠️ Colab Limitations

- 12-hour session limit (training will finish in 4-6 hours, so OK)
- Need to keep browser tab open
- Need to download model after training

## 💡 Alternative: Kaggle Notebooks

Same as Colab but with 30GB RAM:
1. Go to https://www.kaggle.com/
2. Create new notebook
3. Enable GPU
4. Same code as above

---

**Your local machine simply doesn't have enough RAM for this task.** Colab is the fastest, easiest, FREE solution!
