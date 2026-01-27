"""
Centralized Configuration Module for Sakshi AI Surveillance System

This module contains all configuration constants, credentials, and settings
used across the application. Extracted from edit-004.py for better maintainability.

Configuration Sections:
- Device Configuration (CUDA/CPU)
- Frame Processing Settings
- Database Configuration
- File Paths
- API Credentials (Telegram)
- Authentication
- Application Task Configuration
"""

import os
import torch
import pytz
import logging

# =============================================================================
# DEVICE CONFIGURATION (CUDA/CPU)
# =============================================================================

# Enable CUDA auto-detection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configure CUDA optimizations if available
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    logging.info("✅ CUDA ENABLED - Using GPU for processing")
else:
    logging.info("⚠️  CUDA not available - Running in CPU mode")

# =============================================================================
# TIMEZONE CONFIGURATION
# =============================================================================

# Indian Standard Time timezone for all timestamps
IST = pytz.timezone('Asia/Kolkata')

# =============================================================================
# FRAME PROCESSING SETTINGS
# =============================================================================

# Frame downscaling settings to speed up processing/streaming
# Set to None to preserve original camera resolution
TARGET_WIDTH = 640
TARGET_HEIGHT = 360

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# PostgreSQL database connection URL
DATABASE_URL = "postgresql://postgres:Tneural01@127.0.0.1:5432/sakshi"

# =============================================================================
# FILE PATHS
# =============================================================================

# RTSP camera links configuration file
RTSP_LINKS_FILE = 'data/rtsp_links.txt'

# Static files directory for Flask
STATIC_FOLDER = 'static'

# Detections subdirectory within static folder
DETECTIONS_SUBFOLDER = 'detections'

# =============================================================================
# API CREDENTIALS
# =============================================================================

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "7843300957:AAGVv866cPiDPVD0Wrk_wwEEHDSD64Pgaqs"
TELEGRAM_CHAT_ID = "-4835836048"

# =============================================================================
# AUTHENTICATION
# =============================================================================

# Flask application login credentials
LOGIN_USERNAME = "user"
LOGIN_PASSWORD = "Tneural123"

# =============================================================================
# APPLICATION TASK CONFIGURATION
# =============================================================================

# Configuration for each AI detection task/application
# Each task specifies model path, confidence threshold, and other parameters
APP_TASKS_CONFIG = {
    'Generic': {
        'model_path': 'models/02_01_2026_teatost_best.pt',
        'target_class_id': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'confidence': 0.3,
        'is_gif': False
    },
    'PeopleCounter': {
        'model_path': 'models/yolo11n.pt',
        'confidence': 0.15
    },
    'QueueMonitor': {
        'model_path': 'models/yolo11n.pt',
        'confidence': 0.15
    },
    'KitchenCompliance': {
        'model_path': 'models/02_01_2026_teatost_best.pt',
        'confidence': 0.3
    },
    'OccupancyMonitor': {
        'model_path': 'models/yolo11n.pt',
        'confidence': 0.15
    },
    'IdlePeopleViolation': {
        'model_path': 'models/yolo11n.pt',
        'confidence': 0.3
    }
}

# =============================================================================
# DIRECTORY INITIALIZATION
# =============================================================================

# Create necessary directories if they don't exist
os.makedirs(os.path.join(STATIC_FOLDER, DETECTIONS_SUBFOLDER), exist_ok=True)
os.makedirs(os.path.join(STATIC_FOLDER, DETECTIONS_SUBFOLDER, 'shutter_videos'), exist_ok=True)
