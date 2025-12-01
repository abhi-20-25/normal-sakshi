"""
Convert COCO JSON format to YOLO segmentation format for YOLOv11 training
"""
import json
import os
from pathlib import Path
import shutil

def coco_to_yolo_segmentation(coco_json_path, images_dir, output_dir):
    """
    Convert COCO format annotations to YOLO format for segmentation
    
    Args:
        coco_json_path: Path to COCO _annotations.coco.json file
        images_dir: Directory containing the images
        output_dir: Output directory for YOLO format dataset
    """
    
    # Load COCO JSON
    print(f"Loading COCO annotations from: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    output_path = Path(output_dir)
    images_output = output_path / 'images' / 'train'
    labels_output = output_path / 'labels' / 'train'
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Images will be copied to: {images_output}")
    print(f"Labels will be created in: {labels_output}")
    
    # Create category mapping (COCO category_id to YOLO class index)
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    category_names = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    print(f"\nFound {len(categories)} categories:")
    for cat_id, class_idx in categories.items():
        print(f"  Class {class_idx}: {category_names[cat_id]} (COCO ID: {cat_id})")
    
    # Create image_id to filename mapping
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"\nProcessing {len(images_dict)} images...")
    
    processed_count = 0
    for img_id, img_info in images_dict.items():
        filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Copy image to output directory
        src_image = Path(images_dir) / filename
        dst_image = images_output / filename
        
        if src_image.exists():
            shutil.copy2(src_image, dst_image)
        else:
            print(f"Warning: Image not found: {src_image}")
            continue
        
        # Create YOLO format label file
        label_filename = Path(filename).stem + '.txt'
        label_path = labels_output / label_filename
        
        with open(label_path, 'w') as f:
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    class_id = categories[ann['category_id']]
                    segmentation = ann['segmentation']
                    
                    # YOLO format: class_id x1 y1 x2 y2 x3 y3 ...
                    # All coordinates normalized to [0, 1]
                    for seg in segmentation:
                        # Normalize coordinates
                        normalized_coords = []
                        for i in range(0, len(seg), 2):
                            x = seg[i] / width
                            y = seg[i + 1] / height
                            normalized_coords.extend([x, y])
                        
                        # Write to file
                        coords_str = ' '.join([f'{coord:.6f}' for coord in normalized_coords])
                        f.write(f'{class_id} {coords_str}\n')
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} images...")
    
    print(f"\n✅ Conversion complete! Processed {processed_count} images")
    
    # Create data.yaml file for YOLOv11
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"# YOLOv11 Segmentation Dataset\n")
        f.write(f"path: {output_path.absolute()}  # dataset root dir\n")
        f.write(f"train: images/train  # train images (relative to 'path')\n")
        f.write(f"val: images/train  # val images (using train for now)\n\n")
        f.write(f"# Classes\n")
        f.write(f"names:\n")
        for cat_id in sorted(categories.keys()):
            class_idx = categories[cat_id]
            class_name = category_names[cat_id]
            f.write(f"  {class_idx}: {class_name}\n")
    
    print(f"\n✅ Created data.yaml at: {yaml_path}")
    print(f"\nYou can now train YOLOv11 with:")
    print(f"  yolo segment train data={yaml_path} model=yolo11n-seg.pt epochs=100 imgsz=640")
    
    return output_path


if __name__ == "__main__":
    # Define paths
    coco_json = "/home/athul/sakshi/normal-sakshi/teatost.v2-athul.coco-segmentation/train/_annotations.coco.json"
    images_dir = "/home/athul/sakshi/normal-sakshi/teatost.v2-athul.coco-segmentation/train"
    output_dir = "/home/athul/sakshi/normal-sakshi/kitchen_yolo_dataset"
    
    # Convert
    output_path = coco_to_yolo_segmentation(coco_json, images_dir, output_dir)
    
    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("1. Install ultralytics: pip install ultralytics")
    print(f"2. Train: yolo segment train data={output_path}/data.yaml model=yolo11n-seg.pt epochs=100 imgsz=640")
    print("="*60)
