#!/bin/bash

# Linkbuilding post-processing script for Hugo
# This script runs after Hugo build to add links to HTML files

set -e

# Configuration
LINKBUILDER_PATH="${LINKBUILDER_PATH:-themes/boilerplate/linkbuilding/linkbuilder}"
KEYWORDS_DIR="${KEYWORDS_DIR:-data/linkbuilding}"
PUBLIC_DIR="${PUBLIC_DIR:-public}"
LANGUAGES="${LANGUAGES:-en}"
SKIP_PATHS="${SKIP_PATHS:-admin|api|search|tags|categories}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting linkbuilding post-processing...${NC}"

# Check if linkbuilder exists
if [ ! -f "$LINKBUILDER_PATH" ]; then
    echo -e "${YELLOW}Building linkbuilder tool...${NC}"
    cd themes/boilerplate/linkbuilding
    go build -o linkbuilder ./cmd/linkbuilder
    cd - > /dev/null
fi

# Function to process a single HTML file
process_file() {
    local file="$1"
    local lang="$2"
    local keywords_file="${KEYWORDS_DIR}/${lang}.json"
    
    # Skip if keywords file doesn't exist
    if [ ! -f "$keywords_file" ]; then
        return
    fi
    
    # Extract the URL path from the file path
    local url_path="${file#$PUBLIC_DIR}"
    url_path="${url_path%.html}"
    url_path="${url_path%/index}"
    
    # Process the file
    local temp_file="${file}.tmp"
    
    if $LINKBUILDER_PATH -input "$file" -output "$temp_file" -keywords "$keywords_file" -url "$url_path" 2>/dev/null; then
        mv "$temp_file" "$file"
        echo -e "  ✓ Processed: ${file#$PUBLIC_DIR/}"
    else
        rm -f "$temp_file"
        echo -e "  ${RED}✗ Failed: ${file#$PUBLIC_DIR/}${NC}"
    fi
}

# Process each language
for lang in $LANGUAGES; do
    echo -e "${GREEN}Processing language: $lang${NC}"
    
    # Find all HTML files for this language
    lang_dir="$PUBLIC_DIR/$lang"
    
    if [ -d "$lang_dir" ]; then
        # Count files to process
        total_files=$(find "$lang_dir" -name "*.html" -type f | grep -Ev "($SKIP_PATHS)" | wc -l)
        processed=0
        
        echo -e "Found ${total_files} HTML files to process"
        
        # Process each HTML file
        find "$lang_dir" -name "*.html" -type f | grep -Ev "($SKIP_PATHS)" | while read -r file; do
            process_file "$file" "$lang"
            ((processed++)) || true
            
            # Show progress
            if [ $((processed % 10)) -eq 0 ]; then
                echo -e "  Progress: $processed/$total_files"
            fi
        done
    else
        echo -e "${YELLOW}Language directory not found: $lang_dir${NC}"
    fi
done

echo -e "${GREEN}Linkbuilding post-processing complete!${NC}"