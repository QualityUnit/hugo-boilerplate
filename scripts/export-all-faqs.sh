#!/bin/bash
#
# Export FAQs from all directories in content/en/
# This script iterates through each subdirectory and generates CSV files with FAQ data.
#

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/export-faqs.py"

# Base content directory
CONTENT_DIR="content/en"

# Output directory for CSV files
OUTPUT_DIR="data/faq_export"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Check if content directory exists
if [ ! -d "$CONTENT_DIR" ]; then
    echo "Error: Content directory not found at $CONTENT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Exporting FAQs from all directories in $CONTENT_DIR..."
echo "Output directory: $OUTPUT_DIR"
echo "---"

# Counter for processed directories
count=0

# Find all subdirectories in content/en/ (not recursively, just first level)
for dir in "$CONTENT_DIR"/*/ ; do
    if [ -d "$dir" ]; then
        # Get the directory name
        dir_name=$(basename "$dir")

        # Skip hidden directories
        if [[ "$dir_name" == .* ]]; then
            continue
        fi

        # Output CSV file path
        output_file="$OUTPUT_DIR/${dir_name}_faqs.csv"

        echo "Processing: $dir_name"

        # Run the Python script
        python3 "$PYTHON_SCRIPT" "$dir" "$output_file"

        count=$((count + 1))
    fi
done

echo "---"
echo "Processed $count directories"
echo "CSV files saved to: $OUTPUT_DIR/"
