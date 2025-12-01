"""
Complete YOLOv11 Kitchen Compliance Model Training
Single unified model for: Person Detection + Uniform + PPE Violations
"""
from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import gc
import os

# Set environment variables to prevent CUDA errors
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Disable cuDNN benchmarking (can cause cuDNN errors)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train_kitchen_compliance_model():
    """
    Train unified kitchen compliance detection model
    """
    
    print("="*70)
    print("🏭 KITCHEN COMPLIANCE MODEL - YOLOv11 SEGMENTATION TRAINING")
    print("="*70)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🔧 Device Configuration:")
    print(f"   Using: {device.upper()}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        print(f"   GPU cache cleared")
    else:
        print(f"   ⚠️  Training on CPU (will be slower)")
    
    # Load dataset info
    data_yaml = Path('/home/athul/sakshi/normal-sakshi/kitchen_unified_dataset/data.yaml')
    print(f"\n📊 Dataset: {data_yaml}")
    
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\n🎯 Model Classes ({data_config['nc']} total):")
    print("   Base:")
    print("   → 0: person (human detection)")
    print("\n   ✅ Compliant:")
    print("   → 1: uniform (any color)")
    print("   → 3: cap_present")
    print("   → 5: with_apron")
    print("\n   ❌ Violations:")
    print("   → 2: without_uniform")
    print("   → 4: without_cap")
    print("   → 6: without_apron")
    print("   → 7: without_gloves")
    print("   → 8: using_phone")
    
    # Load pretrained model
    print(f"\n📦 Loading YOLOv11n-seg pretrained model...")
    print("   (Includes COCO person detection as base)")
    model = YOLO('yolo11n-seg.pt')  # Nano - fastest
    # Alternatives: yolo11s-seg.pt, yolo11m-seg.pt, yolo11l-seg.pt
    
    print(f"\n🏋️  Starting Training...")
    print(f"   Epochs: 100")
    print(f"   Image size: 320 (ultra-conservative)")
    print(f"   Batch size: 1")
    print(f"   Workers: 1 (single worker to avoid multiprocessing issues)")
    print(f"   🔧 cuDNN deterministic mode enabled")
    print(f"   🔧 Memory optimizations active")
    
    try:
        # Training configuration - ULTRA-CONSERVATIVE FOR cuDNN STABILITY
        results = model.train(
            # Dataset
            data=str(data_yaml),
            epochs=100,
            imgsz=320,  # Further reduced for cuDNN stability
            batch=1,
            
            # Device
            device=device,
            workers=1,  # Single worker to avoid multiprocessing errors
            
            # Output
            project='kitchen_compliance_model',
            name='yolo11n_unified',
            exist_ok=True,
            
            # Training
            patience=30,
            save=True,
            save_period=25,
            cache=False,
            
            # Data Augmentation (DISABLED)
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.0,  # Disabled all augmentation
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            
            # Optimizer
            optimizer='SGD',  # SGD is more stable
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Loss weights
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # Validation
            val=True,
            plots=True,
            
            # Advanced - CRITICAL FOR cuDNN STABILITY
            verbose=True,
            seed=42,
            deterministic=True,  # Force deterministic for stability
            single_cls=False,
            rect=False,
            cos_lr=True,
            close_mosaic=20,
            amp=False,  # DISABLED - can cause cuDNN errors
            fraction=1.0,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            
            # Memory optimization
            max_det=100,
        )
        
    except RuntimeError as e:
        error_str = str(e)
        print(f"\n❌ CUDA/cuDNN Error occurred: {e}")
        print("\n🔧 SOLUTIONS:")
        
        if "cuDNN" in error_str:
            print("cuDNN ERROR DETECTED:")
            print("1. Try training on CPU (slower but stable):")
            print("   Change: device='cpu'")
            print("2. Update CUDA and cuDNN:")
            print("   sudo apt-get update")
            print("   sudo apt-get install --reinstall nvidia-cuda-toolkit")
            print("3. Reinstall PyTorch with matching CUDA:")
            print("   pip uninstall torch torchvision")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("4. Clear GPU memory:")
            print("   nvidia-smi")
            print("   sudo fuser -v /dev/nvidia*")
            print("   sudo kill -9 <PID>")
        
        print("\n5. RECOMMENDED: Use Google Colab with T4 GPU (FREE & STABLE)")
        print("   - No driver issues")
        print("   - 16GB GPU RAM")
        print("   - Properly configured environment")
        raise
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\n📁 Results: kitchen_compliance_model/yolo11n_unified/")
    print(f"🏆 Best model: kitchen_compliance_model/yolo11n_unified/weights/best.pt")
    print(f"📊 Last model: kitchen_compliance_model/yolo11n_unified/weights/last.pt")
    
    # Validate
    print("\n🔍 Running Final Validation...")
    metrics = model.val()
    
    print("\n📊 Validation Metrics:")
    print(f"   Box mAP50: {metrics.box.map50:.4f}")
    print(f"   Box mAP50-95: {metrics.box.map:.4f}")
    print(f"   Mask mAP50: {metrics.seg.map50:.4f}")
    print(f"   Mask mAP50-95: {metrics.seg.map:.4f}")
    
    # Export model (optional)
    print("\n📦 Exporting model formats...")
    try:
        # Export to ONNX for production
        model.export(format='onnx', simplify=True)
        print("   ✅ ONNX export successful")
    except Exception as e:
        print(f"   ⚠️  ONNX export failed: {e}")
    
    print("\n" + "="*70)
    print("🎉 ALL DONE!")
    print("="*70)
    print("\n📝 Next Steps:")
    print("1. Check training plots: kitchen_compliance_model/yolo11n_unified/")
    print("2. Test model: python3 test_kitchen_model.py")
    print("3. Integrate: Use kitchen_compliance_model/yolo11n_unified/weights/best.pt")
    print("="*70)
    
    return model, results, metrics


if __name__ == "__main__":
    import os
    os.chdir('/home/athul/sakshi/normal-sakshi')
    
    # Clear memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 GPU memory cleared")
    
    print("\n🚀 Starting Kitchen Compliance Model Training...")
    print("⏱️  This will take several hours depending on your GPU")
    print("💡 Tip: Monitor progress in kitchen_compliance_model/yolo11n_unified/")
    print("\n⚠️  IMPORTANT: If you get CUDA errors:")
    print("   1. Make sure no other programs are using the GPU")
    print("   2. Run: nvidia-smi to check GPU memory")
    print("   3. Consider training on Google Colab with T4 GPU\n")
    
    try:
        model, results, metrics = train_kitchen_compliance_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Progress saved in: kitchen_compliance_model/yolo11n_unified/weights/last.pt")
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        print("\nCheck the error above and try the suggested solutions")
        raise
