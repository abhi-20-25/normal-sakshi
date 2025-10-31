#!/usr/bin/env python3
"""
WSGI entry point for Gunicorn
This file exposes the Flask app and SocketIO instance to Gunicorn
"""

import sys
import os
import importlib.util

# Get the path to edit-004.py (handle hyphen in filename)
script_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(script_dir, 'edit-004.py')

# Load the module dynamically to handle hyphen in filename
spec = importlib.util.spec_from_file_location("edit_004", module_path)
edit_004 = importlib.util.module_from_spec(spec)
sys.modules["edit_004"] = edit_004
spec.loader.exec_module(edit_004)

# Import app and socketio from the loaded module
app = edit_004.app
socketio = edit_004.socketio

# Gunicorn expects 'application' variable
# For Flask-SocketIO, we need to use socketio as WSGI app
application = socketio.WSGIApp(socketio, app)

# Export application for gunicorn
__all__ = ['application']

