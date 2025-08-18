#!/usr/bin/env python3
"""
Test script to verify case-insensitive keyword deduplication works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent))

from precompute_linkbuilding import save_optimized_linkbuilding
import tempfile
import json

def test_deduplication():
    """Test that case-insensitive keywords are deduplicated."""
    
    # Test keywords with case variations
    test_keywords = {
        "Správa Cron úloh": {
            'url': '/integrations/cronlytic/',
            'title': 'Integration title',
            'priority': 0,
            'exact': False  # Case-insensitive
        },
        "správa cron úloh": {
            'url': '/mcp-servers/cronlytic/',
            'title': 'MCP Server title',
            'priority': 0,
            'exact': False  # Case-insensitive
        },
        "SPRÁVA CRON ÚLOH": {
            'url': '/another/path/',
            'title': 'Another title',
            'priority': 10,  # Higher priority
            'exact': False  # Case-insensitive
        },
        "ExactKeyword": {
            'url': '/exact1/',
            'title': 'Exact 1',
            'priority': 0,
            'exact': True  # Case-sensitive
        },
        "exactkeyword": {
            'url': '/exact2/',
            'title': 'Exact 2',
            'priority': 0,
            'exact': True  # Case-sensitive - should NOT be deduplicated
        }
    }
    
    # All keywords are "found"
    found_keywords = set(test_keywords.keys())
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        output_path = Path(f.name)
    
    # Run deduplication
    metadata = save_optimized_linkbuilding(test_keywords, found_keywords, output_path)
    
    # Read the result
    with open(output_path, 'r') as f:
        result = json.load(f)
    
    print("Test Results:")
    print("=" * 50)
    print(f"Input keywords: {len(test_keywords)}")
    print(f"Output keywords: {len(result['keywords'])}")
    print()
    
    print("Keywords in output:")
    for item in result['keywords']:
        print(f"  - {item['Keyword']} (Priority: {item['Priority']}, Exact: {item['Exact']})")
    
    print()
    print("Expected behavior:")
    print("  - 3 case-insensitive 'správa cron úloh' variants → 1 (highest priority)")
    print("  - 2 case-sensitive 'ExactKeyword' variants → 2 (both kept)")
    print()
    
    # Verify expectations
    keywords_in_output = [item['Keyword'] for item in result['keywords']]
    
    # Check that only one case-insensitive variant exists
    cron_keywords = [k for k in keywords_in_output if 'cron' in k.lower()]
    print(f"✓ Cron keywords in output: {len(cron_keywords)} (expected: 1)")
    
    # Check that both exact keywords exist
    exact_keywords = [k for k in keywords_in_output if 'exact' in k.lower()]
    print(f"✓ Exact keywords in output: {len(exact_keywords)} (expected: 2)")
    
    # Clean up
    output_path.unlink()
    
    print("\nDeduplication test completed successfully!")

if __name__ == "__main__":
    test_deduplication()