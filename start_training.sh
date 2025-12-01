#!/bin/bash

# Kitchen Compliance Model - Training Launcher
# Complete automation script

echo "======================================================================"
echo "🏭 KITCHEN COMPLIANCE MODEL - YOLOv11 TRAINING"
echo "======================================================================"
echo ""

# Check if dataset exists
if [ ! -d "kitchen_unified_dataset" ]; then
    echo "❌ Error: Dataset not found!"
    echo "   Run: python3 create_unified_dataset.py first"
    exit 1
fi

# Check dataset
IMAGES=$(ls kitchen_unified_dataset/images/train/*.jpg 2>/dev/null | wc -l)
LABELS=$(ls kitchen_unified_dataset/labels/train/*.txt 2>/dev/null | wc -l)

echo "📊 Dataset Check:"
echo "   Images: $IMAGES"
echo "   Labels: $LABELS"

if [ "$IMAGES" -eq 0 ] || [ "$LABELS" -eq 0 ]; then
    echo "❌ Error: Dataset is empty!"
    exit 1
fi

echo "   ✅ Dataset OK"
echo ""

# Check GPU
echo "🔧 Hardware Check:"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "   GPU: $GPU_NAME"
    echo "   Memory: $GPU_MEM"
    echo "   ✅ GPU Available"
else
    echo "   ⚠️  No GPU detected - training will be SLOW on CPU"
    echo "   Recommended: Use GPU for training"
    read -p "   Continue with CPU? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ❌ ultralytics not installed"
    echo "   Installing..."
    pip install ultralytics -q
fi

python3 -c "import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ❌ opencv-python not installed"
    echo "   Installing..."
    pip install opencv-python -q
fi

echo "   ✅ Dependencies OK"
echo ""

# Confirm training
echo "======================================================================"
echo "⏱️  TRAINING DETAILS"
echo "======================================================================"
echo "   Model: YOLOv11n-seg (nano)"
echo "   Dataset: 4,530 images, 9 classes"
echo "   Epochs: 150"
echo "   Estimated time: 4-6 hours (GPU) or 24-48 hours (CPU)"
echo ""
echo "   Classes:"
echo "   → 0: person (human detection)"
echo "   → 1: uniform ✅"
echo "   → 2: without_uniform ❌"
echo "   → 3: cap_present ✅"
echo "   → 4: without_cap ❌"
echo "   → 5: with_apron ✅"
echo "   → 6: without_apron ❌"
echo "   → 7: without_gloves ❌"
echo "   → 8: using_phone ❌"
echo ""
echo "======================================================================"

read -p "🚀 Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled"
    exit 0
fi

echo ""
echo "======================================================================"
echo "🏋️  TRAINING STARTED"
echo "======================================================================"
echo ""

# Start training
python3 train_kitchen_unified.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ TRAINING COMPLETED SUCCESSFULLY!"
    echo "======================================================================"
    echo ""
    echo "📁 Model saved to: kitchen_compliance_model/yolo11n_unified/weights/"
    echo ""
    echo "Next steps:"
    echo "1. Check results: kitchen_compliance_model/yolo11n_unified/"
    echo "2. Test model: python3 test_kitchen_model.py"
    echo "3. View training plots and metrics"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "❌ TRAINING FAILED"
    echo "======================================================================"
    echo ""
    echo "Check error messages above"
    echo "Common fixes:"
    echo "- Reduce batch size if GPU out of memory"
    echo "- Ensure dataset is properly formatted"
    echo "- Check CUDA drivers if using GPU"
    echo ""
fi
