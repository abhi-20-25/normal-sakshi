#!/usr/bin/env python3
"""
WSGI entry point for Gunicorn
This file exposes the Flask app and SocketIO instance to Gunicorn
"""

import sys
import os
import importlib.util
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the path to edit-004.py (handle hyphen in filename)
script_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(script_dir, 'edit-004.py')

logger.info(f"Loading application from: {module_path}")

# Load the module dynamically to handle hyphen in filename
spec = importlib.util.spec_from_file_location("edit_004", module_path)
edit_004 = importlib.util.module_from_spec(spec)
sys.modules["edit_004"] = edit_004
spec.loader.exec_module(edit_004)

logger.info("Module loaded successfully")

# Import app and socketio from the loaded module
app = edit_004.app
socketio = edit_004.socketio

# Ensure initialization happens (it should have started on import in background)
# Don't call initialize_app() directly here - it's already running in background thread
# Just verify it's initialized or starting
if hasattr(edit_004, '_initialized'):
    if edit_004._initialized:
        logger.info("Application already initialized")
    elif hasattr(edit_004, '_initialization_thread') and edit_004._initialization_thread and edit_004._initialization_thread.is_alive():
        logger.info("Application initialization in progress (background thread)")
    else:
        logger.warning("Application initialization not started - triggering now")
        if hasattr(edit_004, '_ensure_initialized'):
            edit_004._ensure_initialized(background=True)

# Gunicorn expects 'application' variable
# For Flask-SocketIO, we need to use socketio as WSGI app
application = socketio.WSGIApp(socketio, app)

logger.info("WSGI application ready")

# Export application for gunicorn
__all__ = ['application']

