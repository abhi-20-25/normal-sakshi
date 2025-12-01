"""
Train YOLOv11 Segmentation Model for Kitchen Compliance
"""
from ultralytics import YOLO
import torch

def train_yolo11_segmentation():
    """
    Train YOLOv11 segmentation model on kitchen compliance dataset
    """
    
    print("="*60)
    print("YOLOv11 Segmentation Training - Kitchen Compliance")
    print("="*60)
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🔧 Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load a pretrained YOLOv11 segmentation model
    print("\n📦 Loading YOLOv11n-seg pretrained model...")
    model = YOLO('yolo11n-seg.pt')  # nano model (fastest, smallest)
    # Other options:
    # model = YOLO('yolo11s-seg.pt')  # small
    # model = YOLO('yolo11m-seg.pt')  # medium
    # model = YOLO('yolo11l-seg.pt')  # large
    # model = YOLO('yolo11x-seg.pt')  # extra large
    
    print("\n🏋️ Starting training...")
    print(f"   Dataset: kitchen_yolo_dataset/data.yaml")
    print(f"   Classes: 11 (cafe, uniforms, PPE items)")
    print(f"   Images: 4530 training images")
    
    # Train the model
    results = model.train(
        data='kitchen_yolo_dataset/data.yaml',  # path to data.yaml
        epochs=100,                              # number of epochs
        imgsz=640,                               # image size
        batch=16,                                # batch size (adjust based on GPU memory)
        device=device,                           # device to use
        workers=8,                               # number of workers for dataloader
        project='kitchen_model',                 # save results to kitchen_model/
        name='yolo11n_seg',                      # experiment name
        exist_ok=True,                           # overwrite existing
        patience=20,                             # early stopping patience
        save=True,                               # save checkpoints
        save_period=10,                          # save checkpoint every 10 epochs
        cache=False,                             # cache images for faster training (uses more RAM)
        
        # Augmentation settings
        hsv_h=0.015,                             # image HSV-Hue augmentation
        hsv_s=0.7,                               # image HSV-Saturation augmentation
        hsv_v=0.4,                               # image HSV-Value augmentation
        degrees=0.0,                             # rotation (+/- deg)
        translate=0.1,                           # translation (+/- fraction)
        scale=0.5,                               # scale (+/- gain)
        shear=0.0,                               # shear (+/- deg)
        perspective=0.0,                         # perspective (+/- fraction)
        flipud=0.0,                              # flip up-down (probability)
        fliplr=0.5,                              # flip left-right (probability)
        mosaic=1.0,                              # mosaic augmentation (probability)
        mixup=0.0,                               # mixup augmentation (probability)
        
        # Optimization
        optimizer='auto',                        # optimizer (SGD, Adam, AdamW, auto)
        lr0=0.01,                                # initial learning rate
        lrf=0.01,                                # final learning rate (lr0 * lrf)
        momentum=0.937,                          # SGD momentum/Adam beta1
        weight_decay=0.0005,                     # optimizer weight decay
        warmup_epochs=3.0,                       # warmup epochs
        warmup_momentum=0.8,                     # warmup initial momentum
        warmup_bias_lr=0.1,                      # warmup initial bias lr
        
        # Loss weights
        box=7.5,                                 # box loss gain
        cls=0.5,                                 # cls loss gain
        dfl=1.5,                                 # dfl loss gain
        
        # Validation
        val=True,                                # validate during training
        plots=True,                              # save plots
        
        # Other
        verbose=True,                            # verbose output
        seed=0,                                  # random seed for reproducibility
        deterministic=True,                      # whether to enable deterministic mode
        single_cls=False,                        # train as single-class dataset
        rect=False,                              # rectangular training
        cos_lr=False,                            # use cosine learning rate scheduler
        close_mosaic=10,                         # disable mosaic augmentation for last N epochs
        amp=True,                                # Automatic Mixed Precision training
        fraction=1.0,                            # dataset fraction to train on
        profile=False,                           # profile ONNX and TensorRT speeds
        freeze=None,                             # freeze layers (list of layer indices)
    )
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"\n📊 Results saved to: {model.trainer.save_dir}")
    print(f"🏆 Best model: {model.trainer.best}")
    print(f"📈 Metrics: {model.trainer.metrics}")
    
    # Validate the model
    print("\n🔍 Running final validation...")
    metrics = model.val()
    
    print("\n📊 Validation Metrics:")
    print(f"   mAP50: {metrics.seg.map50:.4f}")
    print(f"   mAP50-95: {metrics.seg.map:.4f}")
    
    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("1. Check training results in kitchen_model/yolo11n_seg/")
    print("2. Best model saved at: kitchen_model/yolo11n_seg/weights/best.pt")
    print("3. Use for inference: model = YOLO('kitchen_model/yolo11n_seg/weights/best.pt')")
    print("="*60)
    
    return model, results


if __name__ == "__main__":
    import os
    os.chdir('/home/athul/sakshi/normal-sakshi')
    
    model, results = train_yolo11_segmentation()
