#!/usr/bin/env python3
"""
Test script to verify Chinese keyword detection works correctly.
"""

import re

def test_keyword_detection():
    # Test Chinese keyword
    chinese_keyword = "21st-dev Magic MCP 服务器"
    test_text = "这是一个包含 21st-dev Magic MCP 服务器 的测试文本。"
    
    print(f"Testing keyword: {chinese_keyword}")
    print(f"Test text: {test_text}")
    print()
    
    # Old approach (broken for Chinese)
    old_pattern = re.compile(r'\b' + re.escape(chinese_keyword.lower()) + r'\b')
    old_match = old_pattern.search(test_text.lower())
    print(f"Old regex (with \\b): Found = {old_match is not None}")
    
    # New approach (works for Chinese)
    if any(ord(char) > 127 for char in chinese_keyword):
        new_pattern = re.compile(re.escape(chinese_keyword.lower()))
        print(f"Using non-Latin pattern (no word boundaries)")
    else:
        new_pattern = re.compile(r'\b' + re.escape(chinese_keyword.lower()) + r'\b')
        print(f"Using Latin pattern (with word boundaries)")
    
    new_match = new_pattern.search(test_text.lower())
    print(f"New regex: Found = {new_match is not None}")
    
    print("\n" + "="*50)
    
    # Test English keyword for comparison
    english_keyword = "MCP Server"
    test_text_en = "This is a test with MCP Server in it."
    
    print(f"\nTesting keyword: {english_keyword}")
    print(f"Test text: {test_text_en}")
    print()
    
    # English should use word boundaries
    if any(ord(char) > 127 for char in english_keyword):
        en_pattern = re.compile(re.escape(english_keyword.lower()))
        print(f"Using non-Latin pattern (no word boundaries)")
    else:
        en_pattern = re.compile(r'\b' + re.escape(english_keyword.lower()) + r'\b')
        print(f"Using Latin pattern (with word boundaries)")
    
    en_match = en_pattern.search(test_text_en.lower())
    print(f"English regex: Found = {en_match is not None}")

if __name__ == "__main__":
    test_keyword_detection()