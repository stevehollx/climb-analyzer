#!/usr/bin/env python3
import sys
import climb_analyzer

print("Python is importing climb_analyzer from:")
print(f"  {climb_analyzer.__file__}")
print()

# Check if it's finding the right file
import os
expected_path = os.path.join(os.getcwd(), "climb_analyzer.py")
actual_path = climb_analyzer.__file__

print(f"Expected path: {expected_path}")
print(f"Actual path:   {actual_path}")
print()

if expected_path == actual_path or expected_path == actual_path.replace('.pyc', '.py'):
    print("✓ Correct file!")
else:
    print("❌ Wrong file! Python is importing from a different location.")
    print()
    print("Fix: Remove the old module from sys.path")

print()
print("Available attributes in module:")
attrs = [a for a in dir(climb_analyzer) if not a.startswith('_')]
for attr in sorted(attrs)[:20]:  # Show first 20
    print(f"  - {attr}")
if len(attrs) > 20:
    print(f"  ... and {len(attrs) - 20} more")
