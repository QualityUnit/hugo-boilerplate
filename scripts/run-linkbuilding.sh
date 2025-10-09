#!/bin/bash

# Linkbuilding runner script for Hugo sites
# This script runs the Python linkbuilding module with proper virtual environment

set -e

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
THEME_DIR="$(dirname "$SCRIPT_DIR")"
HUGO_ROOT="$(dirname "$(dirname "$THEME_DIR")")"

# Virtual environment path
VENV_PATH="$SCRIPT_DIR/venv"

# Python script path
LINKBUILDING_SCRIPT="$SCRIPT_DIR/linkbuilding.py"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    source "$VENV_PATH/bin/activate"
fi

# Check if required modules are installed
if ! python -c "import bs4" 2>/dev/null; then
    echo "Installing required Python packages..."
    pip install beautifulsoup4 lxml
fi

# Default paths
PUBLIC_DIR="$HUGO_ROOT/public"
KEYWORDS_DIR="$HUGO_ROOT/linkbuilding"

# Function to show usage
usage() {
    echo "Usage: $0 [language] [options]"
    echo ""
    echo "Languages:"
    echo "  en    - Process English content (default)"
    echo "  de    - Process German content"
    echo "  es    - Process Spanish content"
    echo "  fr    - Process French content"
    echo "  all   - Process all languages"
    echo ""
    echo "Options:"
    echo "  --dry-run        - Analyze without modifying files"
    echo "  --config FILE    - Use custom config file"
    echo "  --keywords FILE  - Use custom keywords file"
    echo "  --help           - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 en                    # Process English content"
    echo "  $0 de --dry-run          # Dry run for German content"
    echo "  $0 all                   # Process all languages"
    exit 1
}

# Parse arguments
LANGUAGE="${1:-en}"
shift || true

# Handle help
if [ "$LANGUAGE" = "--help" ] || [ "$LANGUAGE" = "-h" ]; then
    usage
fi

# Process based on language
if [ "$LANGUAGE" = "all" ]; then
    echo "Processing all languages..."
    
    # Process English (in root)
    if [ -f "$KEYWORDS_DIR/keywords_en.csv" ]; then
        echo "Processing English content..."
        python "$LINKBUILDING_SCRIPT" \
            -k "$KEYWORDS_DIR/keywords_en.csv" \
            -d "$PUBLIC_DIR" \
            --exclude de es fr it ja ko nl no pl pt ro sk sv tr vi zh ar cs da fi \
            -o "$HUGO_ROOT/linkbuilding-report-en.html" \
            "$@"
    fi
    
    # Process other languages
    for lang in de es fr it ja ko nl no pl pt ro sk sv tr vi zh ar cs da fi; do
        if [ -f "$KEYWORDS_DIR/keywords_${lang}.csv" ] && [ -d "$PUBLIC_DIR/$lang" ]; then
            echo "Processing $lang content..."
            python "$LINKBUILDING_SCRIPT" \
                -k "$KEYWORDS_DIR/keywords_${lang}.csv" \
                -d "$PUBLIC_DIR/$lang" \
                -o "$HUGO_ROOT/linkbuilding-report-${lang}.html" \
                "$@"
        fi
    done
    
    echo "All languages processed!"
    
elif [ "$LANGUAGE" = "en" ]; then
    # Process English content (exclude language subdirectories)
    KEYWORDS_FILE="${KEYWORDS_FILE:-$KEYWORDS_DIR/keywords_en.csv}"
    
    if [ ! -f "$KEYWORDS_FILE" ]; then
        echo "Error: Keywords file not found: $KEYWORDS_FILE"
        echo "Please create the file with your keywords."
        exit 1
    fi
    
    echo "Processing English content..."
    echo "Keywords file: $KEYWORDS_FILE"
    echo "Directory: $PUBLIC_DIR"
    
    python "$LINKBUILDING_SCRIPT" \
        -k "$KEYWORDS_FILE" \
        -d "$PUBLIC_DIR" \
        --exclude de es fr it ja ko nl no pl pt ro sk sv tr vi zh ar cs da fi \
        -o "$HUGO_ROOT/linkbuilding-report-en.html" \
        "$@"
        
else
    # Process specific language
    KEYWORDS_FILE="${KEYWORDS_FILE:-$KEYWORDS_DIR/keywords_${LANGUAGE}.csv}"
    LANG_DIR="$PUBLIC_DIR/$LANGUAGE"
    
    if [ ! -f "$KEYWORDS_FILE" ]; then
        echo "Error: Keywords file not found: $KEYWORDS_FILE"
        echo "Please create the file with your keywords for $LANGUAGE."
        exit 1
    fi
    
    if [ ! -d "$LANG_DIR" ]; then
        echo "Error: Language directory not found: $LANG_DIR"
        echo "Please build the site first: hugo --buildFuture --configDir config_${LANGUAGE}"
        exit 1
    fi
    
    echo "Processing $LANGUAGE content..."
    echo "Keywords file: $KEYWORDS_FILE"
    echo "Directory: $LANG_DIR"
    
    python "$LINKBUILDING_SCRIPT" \
        -k "$KEYWORDS_FILE" \
        -d "$LANG_DIR" \
        -o "$HUGO_ROOT/linkbuilding-report-${LANGUAGE}.html" \
        "$@"
fi

echo "Linkbuilding complete! Check the report file for details."