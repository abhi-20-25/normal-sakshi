# import cv2

# # --- Configuration ---
# # PASTE YOUR RTSP STREAM LINK HERE
# # Example: "rtsp://username:password@ip_address:554/stream_path"
# RTSP_URL = 'rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=4&subtype=0'

# # A global list to store the coordinates of the clicked points
#!/usr/bin/env python3
"""
ROI Coordinate Finder for RTSP Camera
This script helps you find the exact coordinates for ROI areas on your RTSP camera feed.
"""

import cv2
import numpy as np
from shapely.geometry import Polygon

class ROICoordinateFinder:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.roi_points = []
        self.secondary_roi_points = []
        self.current_roi = 'main'  # 'main' or 'secondary'
        self.drawing = False
        self.current_point = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI drawing"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_point = (x, y)
            
            if self.current_roi == 'main':
                self.roi_points.append((x, y))
                print(f"Main ROI point {len(self.roi_points)}: ({x}, {y})")
            else:
                self.secondary_roi_points.append((x, y))
                print(f"Secondary ROI point {len(self.secondary_roi_points)}: ({x}, {y})")
                
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_point = (x, y)
            
    def draw_roi_points(self, frame):
        """Draw ROI points on the frame"""
        display_frame = frame.copy()
        
        # Draw main ROI points (yellow)
        for i, point in enumerate(self.roi_points):
            cv2.circle(display_frame, point, 5, (0, 255, 255), -1)  # Yellow
            cv2.putText(display_frame, f"M{i+1}", (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw secondary ROI points (cyan)
        for i, point in enumerate(self.secondary_roi_points):
            cv2.circle(display_frame, point, 5, (255, 255, 0), -1)  # Cyan
            cv2.putText(display_frame, f"S{i+1}", (point[0]+10, point[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw current point being drawn
        if self.current_point:
            cv2.circle(display_frame, self.current_point, 3, (0, 0, 255), -1)  # Red
        
        # Draw polygons if we have enough points
        if len(self.roi_points) >= 3:
            pts = np.array(self.roi_points, np.int32)
            cv2.polylines(display_frame, [pts], True, (0, 255, 255), 2)  # Yellow polygon
            
        if len(self.secondary_roi_points) >= 3:
            pts = np.array(self.secondary_roi_points, np.int32)
            cv2.polylines(display_frame, [pts], True, (255, 255, 0), 2)  # Cyan polygon
        
        return display_frame
    
    def normalize_coordinates(self, points, frame_width, frame_height):
        """Convert pixel coordinates to normalized coordinates (0-1)"""
        normalized = []
        for x, y in points:
            norm_x = round(x / frame_width, 3)
            norm_y = round(y / frame_height, 3)
            normalized.append([norm_x, norm_y])
        return normalized
    
    def run(self):
        """Main function to capture video and get ROI coordinates"""
        print("ROI Coordinate Finder")
        print("=" * 50)
        print("Instructions:")
        print("1. Click to add points for MAIN ROI (Queue Area) - Yellow")
        print("2. Press 's' to switch to SECONDARY ROI (Cashier Area) - Cyan")
        print("3. Press 'm' to switch back to MAIN ROI")
        print("4. Press 'r' to reset current ROI")
        print("5. Press 'c' to clear all ROIs")
        print("6. Press 'p' to print normalized coordinates")
        print("7. Press 'q' to quit")
        print("=" * 50)
        
        # Connect to RTSP stream
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            print(f"Error: Could not connect to RTSP stream: {self.rtsp_url}")
            return
        
        # Get frame dimensions
        ret, self.frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame from stream")
            return
            
        frame_height, frame_width = self.frame.shape[:2]
        print(f"Frame dimensions: {frame_width}x{frame_height}")
        
        # Create window and set mouse callback
        cv2.namedWindow('ROI Coordinate Finder', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('ROI Coordinate Finder', self.mouse_callback)
        
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Draw ROI points on frame
            display_frame = self.draw_roi_points(self.frame)
            
            # Add instructions on frame
            cv2.putText(display_frame, f"Current ROI: {'MAIN (Queue)' if self.current_roi == 'main' else 'SECONDARY (Cashier)'}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Main ROI points: {len(self.roi_points)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            cv2.putText(display_frame, f"Secondary ROI points: {len(self.secondary_roi_points)}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            cv2.imshow('ROI Coordinate Finder', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.current_roi = 'secondary'
                print("Switched to SECONDARY ROI (Cashier Area)")
            elif key == ord('m'):
                self.current_roi = 'main'
                print("Switched to MAIN ROI (Queue Area)")
            elif key == ord('r'):
                if self.current_roi == 'main':
                    self.roi_points = []
                    print("Reset MAIN ROI")
                else:
                    self.secondary_roi_points = []
                    print("Reset SECONDARY ROI")
            elif key == ord('c'):
                self.roi_points = []
                self.secondary_roi_points = []
                print("Cleared all ROIs")
            elif key == ord('p'):
                self.print_coordinates(frame_width, frame_height)
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def print_coordinates(self, frame_width, frame_height):
        """Print normalized coordinates for both ROIs"""
        print("\n" + "=" * 50)
        print("NORMALIZED COORDINATES FOR YOUR CODE:")
        print("=" * 50)
        
        if self.roi_points:
            main_normalized = self.normalize_coordinates(self.roi_points, frame_width, frame_height)
            print(f'"roi_points": {main_normalized},')
        else:
            print('"roi_points": [],  # No main ROI points defined')
        
        if self.secondary_roi_points:
            secondary_normalized = self.normalize_coordinates(self.secondary_roi_points, frame_width, frame_height)
            print(f'"secondary_roi_points": {secondary_normalized},')
        else:
            print('"secondary_roi_points": [],  # No secondary ROI points defined')
        
        print("=" * 50)
        print("Copy these coordinates to replace lines 61-62 in edit-004.py")
        print("=" * 50)

def main():
    # Replace with your actual RTSP URL
    rtsp_url = "rtsp://admin:cctv%231234@182.65.205.121:554/cam/realmonitor?channel=4&subtype=0"
    
    print(f"Connecting to RTSP stream: {rtsp_url}")
    
    finder = ROICoordinateFinder(rtsp_url)
    finder.run()

if __name__ == "__main__":
    main()