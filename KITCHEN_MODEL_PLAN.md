# Kitchen Compliance Detection - Complete YOLOv11 Model

## 🎯 Project Overview

**Single Unified Model** for kitchen compliance monitoring that detects:
- ✅ **People** (human detection)
- ✅ **Compliant PPE** (uniform, cap, apron)
- ❌ **Violations** (missing PPE, phone usage)

### Model Classes (9 total)

| Class ID | Name | Type | Description |
|----------|------|------|-------------|
| 0 | person | Base | Human detection |
| 1 | uniform | ✅ Compliant | Any uniform (black/white/yellow merged) |
| 2 | without_uniform | ❌ Violation | No uniform worn |
| 3 | cap_present | ✅ Compliant | Wearing chef cap |
| 4 | without_cap | ❌ Violation | No cap |
| 5 | with_apron | ✅ Compliant | Wearing apron |
| 6 | without_apron | ❌ Violation | No apron |
| 7 | without_gloves | ❌ Violation | No gloves |
| 8 | using_phone | ❌ Violation | Using mobile phone |

---

## 📋 Complete Execution Plan

### Step 1: Dataset Preparation ✅ DONE

```bash
cd /home/athul/sakshi/normal-sakshi
python3 create_unified_dataset.py
```

**What it does:**
- ✅ Converts COCO JSON to YOLO format
- ✅ Merges 3 uniform classes → 1 unified class
- ✅ Creates 9-class dataset structure
- ✅ Generates 4,530 training examples

**Output:**
- `kitchen_unified_dataset/`
  - `images/train/` - 4,530 images
  - `labels/train/` - 4,530 YOLO format labels
  - `data.yaml` - Dataset configuration
  - `violation_rules.json` - Violation logic

---

### Step 2: Model Training 🏋️ READY TO START

```bash
cd /home/athul/sakshi/normal-sakshi
python3 train_kitchen_unified.py
```

**Training Configuration:**
- **Base Model:** YOLOv11n-seg (includes COCO person detection)
- **Epochs:** 150
- **Image Size:** 640x640
- **Batch Size:** 16
- **Optimizer:** AdamW
- **Device:** CUDA (GPU) or CPU

**Key Features:**
- ✅ Person detection (from COCO pretrained weights)
- ✅ Data augmentation (mosaic, mixup, flip)
- ✅ Mixed precision training (faster)
- ✅ Early stopping (patience=30)
- ✅ Checkpoint saving every 10 epochs

**Expected Duration:**
- GPU (RTX 3060): ~4-6 hours
- GPU (T4/V100): ~2-4 hours
- CPU: ~24-48 hours ⚠️ Not recommended

**Output:**
- `kitchen_compliance_model/yolo11n_unified/`
  - `weights/best.pt` - Best model
  - `weights/last.pt` - Last epoch
  - Training plots, metrics, confusion matrix

---

### Step 3: Model Testing 🧪

```bash
cd /home/athul/sakshi/normal-sakshi
python3 test_kitchen_model.py
```

**What it does:**
- ✅ Loads trained model
- ✅ Tests on sample images
- ✅ Detects violations automatically
- ✅ Creates visualizations
- ✅ Generates JSON report

**Output:**
- `kitchen_test_results/`
  - Annotated images with detections
  - `test_results.json` - Violation summary

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install ultralytics opencv-python pyyaml
```

### 2. Run Complete Pipeline

```bash
# Already done - dataset created
# python3 create_unified_dataset.py

# Train model (this will take hours)
python3 train_kitchen_unified.py

# Test model
python3 test_kitchen_model.py
```

---

## 📊 Model Architecture

### YOLOv11n-seg Features:
- **Backbone:** CSPDarknet with C2f modules
- **Neck:** PAN (Path Aggregation Network)
- **Head:** Dual heads (detection + segmentation)
- **Parameters:** ~3M (lightweight)
- **Speed:** ~200 FPS (GPU)

### Person Detection:
- Leverages COCO pretrained weights (class 0)
- 80k images of people in various poses
- Robust to occlusion and varying scales

---

## 🎨 Violation Detection Logic

### How it Works:

1. **Person Detection** (class 0)
   - Detects all people in frame
   - Base for tracking individuals

2. **Uniform Check**
   - If `uniform` (1) detected → ✅ Compliant
   - If `without_uniform` (2) detected → ❌ **VIOLATION**

3. **PPE Checks**
   - Cap: If `without_cap` (4) → ❌ **VIOLATION**
   - Apron: If `without_apron` (6) → ❌ **VIOLATION**
   - Gloves: If `without_gloves` (7) → ❌ **VIOLATION**

4. **Behavior Check**
   - If `using_phone` (8) → ❌ **VIOLATION**

### Violation Scoring:
```python
violations = {
    'uniform': class_2_detected,
    'cap': class_4_detected,
    'apron': class_6_detected,
    'gloves': class_7_detected,
    'phone': class_8_detected
}

total_violations = sum(violations.values())
compliance_score = (5 - total_violations) / 5 * 100
```

---

## 🔧 Integration with Existing System

### Replace Multiple Models:

**Before:**
- `apron-cap.pt` (separate)
- `gloves.pt` (separate)
- `security.pt` or custom uniform detector (separate)
- Person detection (separate)

**After:**
- `kitchen_compliance_model/yolo11n_unified/weights/best.pt` (**ONE MODEL**)

### Code Integration:

```python
from ultralytics import YOLO

# Load unified model
model = YOLO('kitchen_compliance_model/yolo11n_unified/weights/best.pt')

# Run detection
results = model.predict(frame, conf=0.3)

# Process results
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        
        if cls == 0:
            # Person detected
            person_count += 1
        elif cls in [2, 4, 6, 7, 8]:
            # Violation detected
            violations.append(class_names[cls])
        elif cls in [1, 3, 5]:
            # Compliant PPE detected
            compliant_items.append(class_names[cls])
```

---

## 📈 Expected Performance

### Metrics (estimated after training):

| Metric | Expected Value |
|--------|---------------|
| mAP50 (Box) | 0.75 - 0.85 |
| mAP50-95 (Box) | 0.45 - 0.60 |
| mAP50 (Mask) | 0.70 - 0.80 |
| mAP50-95 (Mask) | 0.40 - 0.55 |
| Inference Speed (GPU) | ~50-100 FPS |
| Inference Speed (CPU) | ~5-10 FPS |

### Class-specific Performance:
- **Person:** Very high (leverages COCO pretrained)
- **Uniform:** High (merged classes = more data)
- **PPE items:** Good (depends on annotation quality)
- **Phone usage:** Moderate (challenging to detect)

---

## 🔍 Monitoring Training

### Watch Progress:

```bash
# View training logs
tail -f kitchen_compliance_model/yolo11n_unified/train.log

# TensorBoard (if enabled)
tensorboard --logdir kitchen_compliance_model/yolo11n_unified
```

### Check Results:
- `results.png` - Training curves
- `confusion_matrix.png` - Per-class accuracy
- `val_batch0_pred.jpg` - Sample predictions

---

## ⚡ Optimization Tips

### For Better Performance:

1. **Increase Batch Size** (if GPU memory allows)
   ```python
   batch=32  # Instead of 16
   ```

2. **Larger Model** (if accuracy is priority)
   ```python
   model = YOLO('yolo11m-seg.pt')  # Medium instead of nano
   ```

3. **More Epochs** (if not converging)
   ```python
   epochs=200  # Instead of 150
   ```

4. **Data Split** (for better validation)
   - Create separate val split
   - Update `data.yaml` with val path

---

## 🐛 Troubleshooting

### Common Issues:

**GPU Out of Memory:**
```python
batch=8  # Reduce batch size
imgsz=512  # Reduce image size
```

**Slow Training on CPU:**
- Use Google Colab (free GPU)
- Or AWS/Azure GPU instances

**Low Accuracy:**
- Check annotation quality
- Increase epochs
- Try larger model (yolo11s-seg or yolo11m-seg)

**Model Not Detecting:**
- Lower confidence threshold: `conf=0.2`
- Check if correct model loaded
- Verify dataset classes match

---

## 📦 Final Deliverables

After training completes:

1. **Model File:** `kitchen_compliance_model/yolo11n_unified/weights/best.pt`
2. **Test Results:** `kitchen_test_results/`
3. **Training Report:** `kitchen_compliance_model/yolo11n_unified/results.csv`
4. **Visualizations:** Plots and confusion matrices

---

## 🎉 Next Steps After Training

1. **Validate Performance:**
   - Test on real kitchen footage
   - Measure FPS on target hardware
   - Tune confidence thresholds

2. **Integrate into Production:**
   - Replace existing models in `edit-004.py`
   - Update KitchenComplianceProcessor class
   - Test end-to-end pipeline

3. **Monitor and Improve:**
   - Collect edge cases
   - Retrain with additional data
   - Fine-tune for specific violations

---

## 📞 Support

For issues or questions:
1. Check training logs
2. Review error messages
3. Verify dataset structure
4. Test with smaller batch size

---

## ✅ Checklist

- [x] Dataset created (4,530 images)
- [x] Classes unified (11 → 9)
- [x] Training script ready
- [x] Test script ready
- [ ] **Start training** ← YOU ARE HERE
- [ ] Validate results
- [ ] Integrate into production

---

**Ready to train? Run:**
```bash
python3 train_kitchen_unified.py
```

**⏱️ Estimated time: 4-6 hours (GPU) or 24-48 hours (CPU)**
