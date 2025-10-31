# Gunicorn configuration file for Flask-SocketIO application
# This config enables async workers needed for Flask-SocketIO and video streaming

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:5001"
backlog = 2048

# Worker processes
# Use eventlet workers for Flask-SocketIO support
# Install eventlet: pip install eventlet
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "eventlet"
worker_connections = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Process naming
proc_name = "sakshi-ai-server"

# Server mechanics
daemon = False
pidfile = "/var/run/gunicorn/sakshi-ai.pid"
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = None
# certfile = None

# Performance
max_requests = 1000
max_requests_jitter = 50

# Graceful timeout for worker shutdown
graceful_timeout = 30

# Application
# Gunicorn will import your application
# Make sure edit-004.py exposes 'app' and 'socketio' as module-level variables

