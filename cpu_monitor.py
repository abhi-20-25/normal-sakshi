#!/usr/bin/env python3
"""
CPU Monitor Script
Monitors CPU usage per core every 50 seconds and logs to JSON file
"""

import psutil
import json
import time
from datetime import datetime
import os

LOG_FILE = "cpu_usage_log.json"
INTERVAL_SECONDS = 50

def get_cpu_stats():
    """Get detailed CPU statistics"""
    # Get per-core CPU percentages
    cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)
    
    # Get overall CPU stats
    cpu_times = psutil.cpu_times_percent(interval=0)
    
    # Get load average (1, 5, 15 minutes)
    load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
    
    # Get memory info
    memory = psutil.virtual_memory()
    
    # Get CPU frequency
    cpu_freq = psutil.cpu_freq()
    
    stats = {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "overall_percent": psutil.cpu_percent(interval=0),
            "core_count": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
            "per_core_percent": {
                f"core_{i}": round(percent, 2) 
                for i, percent in enumerate(cpu_percent_per_core)
            },
            "user_percent": round(cpu_times.user, 2),
            "system_percent": round(cpu_times.system, 2),
            "idle_percent": round(cpu_times.idle, 2),
            "frequency_mhz": round(cpu_freq.current, 2) if cpu_freq else None,
            "load_average": {
                "1_min": round(load_avg[0], 2),
                "5_min": round(load_avg[1], 2),
                "15_min": round(load_avg[2], 2)
            }
        },
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent": round(memory.percent, 2)
        },
        "process_count": len(psutil.pids())
    }
    
    return stats

def load_existing_logs():
    """Load existing log file if it exists"""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_logs(logs):
    """Save logs to JSON file"""
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

def print_stats(stats):
    """Print formatted stats to console"""
    print(f"\n{'='*60}")
    print(f"Timestamp: {stats['timestamp']}")
    print(f"{'='*60}")
    print(f"Overall CPU Usage: {stats['cpu']['overall_percent']}%")
    print(f"CPU Cores: {stats['cpu']['core_count']} (Physical: {stats['cpu']['physical_cores']})")
    print(f"\nPer-Core Usage:")
    for core, percent in stats['cpu']['per_core_percent'].items():
        bar = '█' * int(percent / 5) + '░' * (20 - int(percent / 5))
        print(f"  {core:8s}: {bar} {percent:5.1f}%")
    
    print(f"\nCPU Time Distribution:")
    print(f"  User:   {stats['cpu']['user_percent']}%")
    print(f"  System: {stats['cpu']['system_percent']}%")
    print(f"  Idle:   {stats['cpu']['idle_percent']}%")
    
    print(f"\nLoad Average: {stats['cpu']['load_average']['1_min']}, "
          f"{stats['cpu']['load_average']['5_min']}, "
          f"{stats['cpu']['load_average']['15_min']}")
    
    print(f"\nMemory: {stats['memory']['used_gb']}GB / {stats['memory']['total_gb']}GB "
          f"({stats['memory']['percent']}%)")
    
    print(f"Processes: {stats['process_count']}")
    print(f"{'='*60}\n")

def main():
    """Main monitoring loop"""
    print(f"CPU Monitor Started")
    print(f"Logging to: {LOG_FILE}")
    print(f"Check interval: {INTERVAL_SECONDS} seconds")
    print(f"Press Ctrl+C to stop\n")
    
    logs = load_existing_logs()
    
    try:
        while True:
            # Get stats
            stats = get_cpu_stats()
            
            # Print to console
            print_stats(stats)
            
            # Add to logs
            logs.append(stats)
            
            # Keep only last 1000 entries (to prevent file from growing too large)
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Save to file
            save_logs(logs)
            print(f"Logged to {LOG_FILE} (Total entries: {len(logs)})")
            
            # Wait for next interval
            time.sleep(INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        print(f"\n\nMonitoring stopped. Total logs: {len(logs)}")
        print(f"Log file: {LOG_FILE}")

if __name__ == "__main__":
    main()
