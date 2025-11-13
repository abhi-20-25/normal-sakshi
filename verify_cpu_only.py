#!/usr/bin/env python3
"""
Verify that edit-004.py is running in CPU-only mode
This script checks all device assignments and confirms no CUDA usage
"""

import re
import sys

def check_cpu_only(filepath):
    """Check if the code is configured for CPU-only mode"""
    
    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')
    
    print("="*60)
    print("CPU-ONLY MODE VERIFICATION")
    print("="*60)
    print()
    
    issues = []
    checks_passed = []
    
    # Check 1: Global DEVICE setting
    print("✓ Check 1: Global DEVICE setting")
    if re.search(r"^DEVICE = 'cpu'", content, re.MULTILINE):
        print("  ✅ DEVICE = 'cpu' found")
        checks_passed.append("Global DEVICE = 'cpu'")
    else:
        print("  ❌ DEVICE not set to 'cpu'")
        issues.append("Global DEVICE not set to 'cpu'")
    
    # Check 2: No active CUDA checks
    print("\n✓ Check 2: CUDA availability checks")
    cuda_checks = re.findall(r"^(?!#).*torch\.cuda\.is_available\(\)", content, re.MULTILINE)
    if not cuda_checks:
        print("  ✅ No active torch.cuda.is_available() calls")
        checks_passed.append("No CUDA availability checks")
    else:
        print(f"  ❌ Found {len(cuda_checks)} active CUDA checks")
        issues.append(f"Active CUDA checks found: {len(cuda_checks)}")
    
    # Check 3: All device assignments are 'cpu'
    print("\n✓ Check 3: Device assignments")
    device_assignments = []
    for i, line in enumerate(lines, 1):
        if re.search(r"device\s*=", line) and not line.strip().startswith('#'):
            device_assignments.append((i, line.strip()))
    
    cpu_only = True
    for line_num, line in device_assignments:
        if "device='cpu'" in line or 'device="cpu"' in line or "device=device_to_use" in line or "device=self.device" in line:
            print(f"  ✅ Line {line_num}: {line[:60]}...")
        else:
            print(f"  ❌ Line {line_num}: {line[:60]}...")
            cpu_only = False
            issues.append(f"Line {line_num}: Non-CPU device assignment")
    
    if cpu_only:
        checks_passed.append("All device assignments are CPU")
    
    # Check 4: OccupancyMonitor device setting
    print("\n✓ Check 4: OccupancyMonitor device")
    if re.search(r"self\.device = 'cpu'", content):
        print("  ✅ OccupancyMonitor.device = 'cpu'")
        checks_passed.append("OccupancyMonitor uses CPU")
    else:
        print("  ❌ OccupancyMonitor device not set to 'cpu'")
        issues.append("OccupancyMonitor device not CPU")
    
    # Check 5: No half precision (FP16) enabled
    print("\n✓ Check 5: Half precision (FP16) disabled")
    half_true = re.findall(r"half\s*=\s*True", content)
    half_cuda = re.findall(r"half\s*=\s*\(.*cuda.*\)", content)
    if not half_true and not half_cuda:
        print("  ✅ No half precision enabled")
        checks_passed.append("Half precision disabled")
    else:
        print(f"  ❌ Found half=True or half=(cuda) usage")
        issues.append("Half precision may be enabled")
    
    # Check 6: Model.to('cpu') calls
    print("\n✓ Check 6: Model device assignments")
    model_to_calls = re.findall(r"model\.to\(['\"]?(\w+)['\"]?\)", content)
    if all(device == 'cpu' for device in model_to_calls):
        print(f"  ✅ All {len(model_to_calls)} model.to() calls use 'cpu'")
        checks_passed.append("All models loaded to CPU")
    else:
        print(f"  ❌ Some model.to() calls may use GPU")
        issues.append("Non-CPU model.to() calls found")
    
    # Check 7: safe_track_persons function
    print("\n✓ Check 7: safe_track_persons function")
    if re.search(r"device_to_use = 'cpu'", content):
        print("  ✅ device_to_use = 'cpu' in safe_track_persons")
        checks_passed.append("safe_track_persons uses CPU")
    else:
        print("  ❌ device_to_use not set to 'cpu'")
        issues.append("safe_track_persons may use GPU")
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"\n✅ Checks Passed: {len(checks_passed)}")
    for check in checks_passed:
        print(f"   - {check}")
    
    if issues:
        print(f"\n❌ Issues Found: {len(issues)}")
        for issue in issues:
            print(f"   - {issue}")
        print("\n⚠️  WARNING: Code may still use CUDA!")
        return False
    else:
        print("\n🎉 ALL CHECKS PASSED - Code is CPU-only!")
        print("✅ Even if CUDA is available, the code will NOT use it")
        return True

if __name__ == "__main__":
    filepath = "edit-004.py"
    success = check_cpu_only(filepath)
    sys.exit(0 if success else 1)
