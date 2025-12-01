"""
Test Kitchen Compliance Model - Inference and Violation Detection
"""
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import json

class KitchenComplianceDetector:
    """
    Unified kitchen compliance detector
    Detects people, uniforms, and PPE violations in a single pass
    """
    
    def __init__(self, model_path):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained YOLOv11 model (.pt file)
        """
        print(f"🔧 Loading model: {model_path}")
        self.model = YOLO(model_path)
        
        # Class definitions
        self.class_names = {
            0: 'person',
            1: 'uniform',          # ✅ Compliant
            2: 'without_uniform',  # ❌ Violation
            3: 'cap_present',      # ✅ Compliant
            4: 'without_cap',      # ❌ Violation
            5: 'with_apron',       # ✅ Compliant
            6: 'without_apron',    # ❌ Violation
            7: 'without_gloves',   # ❌ Violation
            8: 'using_phone',      # ❌ Violation
        }
        
        self.compliant_classes = [1, 3, 5]  # uniform, cap, apron
        self.violation_classes = [2, 4, 6, 7, 8]  # all violations
        
        # Colors for visualization
        self.colors = {
            0: (255, 255, 255),   # person - white
            1: (0, 255, 0),       # uniform - green
            2: (0, 0, 255),       # without_uniform - red
            3: (0, 255, 0),       # cap_present - green
            4: (0, 0, 255),       # without_cap - red
            5: (0, 255, 0),       # with_apron - green
            6: (0, 0, 255),       # without_apron - red
            7: (0, 0, 255),       # without_gloves - red
            8: (0, 0, 255),       # using_phone - red
        }
        
        print("✅ Model loaded successfully")
    
    def detect(self, image_path, conf_threshold=0.25):
        """
        Run detection on image
        
        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold for detections
            
        Returns:
            results: YOLO results object
            violations: List of detected violations
        """
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=0.45,
            imgsz=640,
            device='cuda' if self.model.device.type == 'cuda' else 'cpu',
            verbose=False
        )
        
        # Extract violations
        violations = []
        people_count = 0
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls == 0:  # person
                        people_count += 1
                    elif cls in self.violation_classes:
                        violations.append({
                            'type': self.class_names[cls],
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy().tolist()
                        })
        
        return results[0], violations, people_count
    
    def visualize(self, image_path, output_path=None, conf_threshold=0.25):
        """
        Visualize detections on image
        
        Args:
            image_path: Input image path
            output_path: Output image path (optional)
            conf_threshold: Confidence threshold
            
        Returns:
            annotated_image: Image with annotations
            violations: List of violations
        """
        # Run detection
        result, violations, people_count = self.detect(image_path, conf_threshold)
        
        # Read image
        image = cv2.imread(str(image_path))
        annotated_image = image.copy()
        
        # Draw detections
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get color
                color = self.colors.get(cls, (255, 255, 255))
                
                # Draw box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"{self.class_names[cls]}: {conf:.2f}"
                if cls in self.violation_classes:
                    label = f"❌ {label}"
                elif cls in self.compliant_classes:
                    label = f"✅ {label}"
                
                # Draw label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_image, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add summary overlay
        summary_y = 30
        cv2.rectangle(annotated_image, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.putText(annotated_image, f"People: {people_count}", (20, summary_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_image, f"Violations: {len(violations)}", (20, summary_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if violations else (0, 255, 0), 2)
        
        if violations:
            violation_text = ", ".join([v['type'] for v in violations[:3]])
            cv2.putText(annotated_image, violation_text[:40], (20, summary_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(str(output_path), annotated_image)
            print(f"✅ Saved to: {output_path}")
        
        return annotated_image, violations


def test_model(model_path, test_images_dir, output_dir):
    """
    Test model on multiple images
    
    Args:
        model_path: Path to trained model
        test_images_dir: Directory with test images
        output_dir: Output directory for results
    """
    print("="*70)
    print("🧪 TESTING KITCHEN COMPLIANCE MODEL")
    print("="*70)
    
    # Create detector
    detector = KitchenComplianceDetector(model_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get test images
    test_images = list(Path(test_images_dir).glob('*.jpg'))
    test_images += list(Path(test_images_dir).glob('*.png'))
    
    print(f"\n📸 Found {len(test_images)} test images")
    
    # Process images
    all_results = []
    for i, img_path in enumerate(test_images[:20]):  # Test first 20
        print(f"\n[{i+1}/{min(20, len(test_images))}] Processing: {img_path.name}")
        
        # Run detection
        output_img_path = output_path / f"result_{img_path.name}"
        annotated_img, violations = detector.visualize(
            img_path,
            output_img_path,
            conf_threshold=0.3
        )
        
        # Store results
        result = {
            'image': img_path.name,
            'violations': violations,
            'violation_count': len(violations)
        }
        all_results.append(result)
        
        # Print summary
        if violations:
            print(f"   ❌ {len(violations)} violations detected:")
            for v in violations:
                print(f"      - {v['type']} (conf: {v['confidence']:.2f})")
        else:
            print("   ✅ No violations detected")
    
    # Save summary
    summary_path = output_path / 'test_results.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, indent=2, fp=f)
    
    print(f"\n✅ Test complete! Results saved to: {output_path}")
    print(f"📊 Summary: {summary_path}")
    
    # Print overall statistics
    total_violations = sum(r['violation_count'] for r in all_results)
    print(f"\n📈 Overall Statistics:")
    print(f"   Images tested: {len(all_results)}")
    print(f"   Total violations: {total_violations}")
    print(f"   Average violations/image: {total_violations/len(all_results):.2f}")


if __name__ == "__main__":
    # Configuration
    model_path = "kitchen_compliance_model/yolo11n_unified/weights/best.pt"
    test_images_dir = "kitchen_unified_dataset/images/train"
    output_dir = "kitchen_test_results"
    
    # Test
    test_model(model_path, test_images_dir, output_dir)
