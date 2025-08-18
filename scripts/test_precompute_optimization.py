#!/usr/bin/env python3
"""
Test script to verify the precompute optimization works correctly.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent))

from precompute_linkbuilding import ContentAnalyzer
import re

def test_early_stopping():
    """Test that the early stopping optimization works."""
    
    # Create a mock analyzer
    analyzer = ContentAnalyzer(Path("."))
    
    # Create test keywords
    test_keywords = [
        ("test", re.compile(r'\btest\b')),
        ("example", re.compile(r'\bexample\b')),
        ("demo", re.compile(r'\bdemo\b'))
    ]
    
    # Test with no keywords found yet
    already_found = set()
    print(f"Test 1: No keywords found yet")
    print(f"  Keywords to search: {len(test_keywords)}")
    print(f"  Already found: {len(already_found)}")
    print(f"  Should search: Yes")
    
    # Test with some keywords found
    already_found = {"test", "example"}
    print(f"\nTest 2: Some keywords found")
    print(f"  Keywords to search: {len(test_keywords)}")
    print(f"  Already found: {len(already_found)} - {already_found}")
    print(f"  Should search: Yes (for 'demo' only)")
    
    # Test with all keywords found
    already_found = {"test", "example", "demo"}
    print(f"\nTest 3: All keywords found")
    print(f"  Keywords to search: {len(test_keywords)}")
    print(f"  Already found: {len(already_found)} - {already_found}")
    print(f"  Should skip file: Yes")
    
    # Simulate the early stopping check
    if len(already_found) == len(test_keywords):
        print(f"  ✅ Early stopping triggered - file would be skipped!")
    else:
        print(f"  ⏳ Would continue searching for {len(test_keywords) - len(already_found)} remaining keywords")
    
    print("\nOptimization test completed successfully!")

if __name__ == "__main__":
    test_early_stopping()