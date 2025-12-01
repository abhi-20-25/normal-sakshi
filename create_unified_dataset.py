"""
Complete Kitchen Compliance Model - YOLOv11 Segmentation
Combines human detection + uniform detection + PPE compliance in one model

Classes:
- 0: person (from COCO pretrained)
- 1: uniform (any: black/white/yellow) ✅
- 2: without_uniform ❌
- 3: cap_present ✅
- 4: without_cap ❌
- 5: with_apron ✅
- 6: without_apron ❌
- 7: without_gloves ❌
- 8: using_phone ❌
"""
import json
import os
from pathlib import Path
import shutil

def create_unified_kitchen_model_dataset(coco_json_path, images_dir, output_dir):
    """
    Create a unified dataset with:
    - Person detection (class 0)
    - Uniform classes merged (class 1)
    - Violation classes (classes 2-8)
    """
    
    print("="*70)
    print("Creating Unified Kitchen Compliance Dataset for YOLOv11")
    print("="*70)
    
    # Load COCO JSON
    print(f"\n📂 Loading annotations: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    output_path = Path(output_dir)
    images_output = output_path / 'images' / 'train'
    labels_output = output_path / 'labels' / 'train'
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)
    
    # Define class mapping - NEW UNIFIED STRUCTURE
    UNIFIED_CLASSES = {
        'person': 0,           # Will be added from person detection
        'uniform': 1,          # ✅ Merged: uniform_black, uniform_white, uniform_yellow
        'without_uniform': 2,  # ❌ Violation
        'cap_present': 3,      # ✅ Compliant
        'without_cap': 4,      # ❌ Violation
        'with_apron': 5,       # ✅ Compliant
        'without_apron': 6,    # ❌ Violation
        'without_gloves': 7,   # ❌ Violation
        'using_phone': 8,      # ❌ Violation
    }
    
    # Map original COCO categories to unified classes
    category_mapping = {}
    for cat in coco_data['categories']:
        cat_name = cat['name']
        cat_id = cat['id']
        
        # Merge all uniform types into single 'uniform' class
        if cat_name in ['uniform_black', 'uniform_white', 'uniform_yellow']:
            category_mapping[cat_id] = UNIFIED_CLASSES['uniform']
        elif cat_name == 'without_uniform':
            category_mapping[cat_id] = UNIFIED_CLASSES['without_uniform']
        elif cat_name == 'cap_present':
            category_mapping[cat_id] = UNIFIED_CLASSES['cap_present']
        elif cat_name == 'without_cap':
            category_mapping[cat_id] = UNIFIED_CLASSES['without_cap']
        elif cat_name == 'with_apron':
            category_mapping[cat_id] = UNIFIED_CLASSES['with_apron']
        elif cat_name == 'without_apron':
            category_mapping[cat_id] = UNIFIED_CLASSES['without_apron']
        elif cat_name == 'without_gloves':
            category_mapping[cat_id] = UNIFIED_CLASSES['without_gloves']
        elif cat_name == 'using_phone':
            category_mapping[cat_id] = UNIFIED_CLASSES['using_phone']
        # Skip 'cafe' as it's not needed
    
    print(f"\n🔧 Class Mapping:")
    print(f"   Original classes: {len(coco_data['categories'])}")
    print(f"   Unified classes: {len(UNIFIED_CLASSES)}")
    print(f"\n   Merged uniforms: uniform_black + uniform_white + uniform_yellow → uniform (class 1)")
    
    # Create image_id to filename mapping
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"\n📊 Processing {len(images_dict)} images...")
    
    processed_count = 0
    for img_id, img_info in images_dict.items():
        filename = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Copy image
        src_image = Path(images_dir) / filename
        dst_image = images_output / filename
        
        if src_image.exists():
            shutil.copy2(src_image, dst_image)
        else:
            print(f"⚠️  Warning: Image not found: {src_image}")
            continue
        
        # Create YOLO format label file
        label_filename = Path(filename).stem + '.txt'
        label_path = labels_output / label_filename
        
        with open(label_path, 'w') as f:
            if img_id in annotations_by_image:
                for ann in annotations_by_image[img_id]:
                    original_cat_id = ann['category_id']
                    
                    # Skip if category not in mapping (like 'cafe')
                    if original_cat_id not in category_mapping:
                        continue
                    
                    # Get unified class ID
                    class_id = category_mapping[original_cat_id]
                    segmentation = ann['segmentation']
                    
                    # YOLO format: class_id x1 y1 x2 y2 x3 y3 ...
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
        if processed_count % 500 == 0:
            print(f"   ✓ Processed {processed_count} images...")
    
    print(f"\n✅ Conversion complete! Processed {processed_count} images")
    
    # Create data.yaml for YOLOv11
    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write("# Kitchen Compliance Detection - YOLOv11 Segmentation\n")
        f.write("# Detects people + uniforms + PPE violations\n\n")
        f.write(f"path: {output_path.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/train  # Using train for validation (split later if needed)\n\n")
        f.write("# Number of classes\n")
        f.write(f"nc: {len(UNIFIED_CLASSES)}\n\n")
        f.write("# Class names\n")
        f.write("names:\n")
        f.write("  0: person           # Human detection (base)\n")
        f.write("  1: uniform          # ✅ Any uniform (black/white/yellow)\n")
        f.write("  2: without_uniform  # ❌ VIOLATION\n")
        f.write("  3: cap_present      # ✅ Wearing cap\n")
        f.write("  4: without_cap      # ❌ VIOLATION\n")
        f.write("  5: with_apron       # ✅ Wearing apron\n")
        f.write("  6: without_apron    # ❌ VIOLATION\n")
        f.write("  7: without_gloves   # ❌ VIOLATION\n")
        f.write("  8: using_phone      # ❌ VIOLATION\n")
    
    print(f"\n✅ Created data.yaml at: {yaml_path}")
    
    # Create violation rules file
    rules_path = output_path / 'violation_rules.json'
    rules = {
        "compliant_classes": [1, 3, 5],  # uniform, cap_present, with_apron
        "violation_classes": [2, 4, 6, 7, 8],  # all without_* and using_phone
        "class_descriptions": {
            "0": "person - base human detection",
            "1": "uniform - wearing proper uniform (any color)",
            "2": "without_uniform - VIOLATION: no uniform",
            "3": "cap_present - wearing chef cap/hat",
            "4": "without_cap - VIOLATION: no cap",
            "5": "with_apron - wearing apron",
            "6": "without_apron - VIOLATION: no apron",
            "7": "without_gloves - VIOLATION: no gloves",
            "8": "using_phone - VIOLATION: using mobile phone"
        },
        "violation_logic": {
            "uniform_check": "If class 2 (without_uniform) detected → VIOLATION",
            "cap_check": "If class 4 (without_cap) detected → VIOLATION",
            "apron_check": "If class 6 (without_apron) detected → VIOLATION",
            "gloves_check": "If class 7 (without_gloves) detected → VIOLATION",
            "phone_check": "If class 8 (using_phone) detected → VIOLATION"
        }
    }
    
    with open(rules_path, 'w') as f:
        json.dump(rules, indent=2, fp=f)
    
    print(f"✅ Created violation_rules.json at: {rules_path}")
    
    return output_path, UNIFIED_CLASSES


if __name__ == "__main__":
    # Paths
    coco_json = "/home/athul/sakshi/normal-sakshi/teatost.v2-athul.coco-segmentation/train/_annotations.coco.json"
    images_dir = "/home/athul/sakshi/normal-sakshi/teatost.v2-athul.coco-segmentation/train"
    output_dir = "/home/athul/sakshi/normal-sakshi/kitchen_unified_dataset"
    
    # Create dataset
    dataset_path, classes = create_unified_kitchen_model_dataset(coco_json, images_dir, output_dir)
    
    print("\n" + "="*70)
    print("✅ UNIFIED DATASET CREATED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📁 Dataset location: {dataset_path}")
    print(f"📊 Total classes: {len(classes)}")
    print(f"📷 Images: {len(list((dataset_path / 'images' / 'train').glob('*.jpg')))}")
    print(f"🏷️  Labels: {len(list((dataset_path / 'labels' / 'train').glob('*.txt')))}")
    
    print("\n🎯 Class Structure:")
    print("   Base Detection:")
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
    
    print("\n" + "="*70)
    print("Next: Run training script")
    print("="*70)
