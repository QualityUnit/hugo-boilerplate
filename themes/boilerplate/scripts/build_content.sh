#!/bin/bash
# build_content.sh
#
# This script creates a virtual environment, installs requirements,
# and performs two main tasks:
# 1. Translates missing content files from English to other languages
# 2. Generates related content YAML files for the Hugo site

set -e  # Exit immediately if a command exits with a non-zero status

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
THEME_DIR="$(dirname "$SCRIPT_DIR")"
HUGO_ROOT="$(dirname "$(dirname "$THEME_DIR")")"

#print directories

echo -e "${BLUE}=== Directories ===${NC}"
echo -e "${BLUE}Script directory: ${SCRIPT_DIR}${NC}"
echo -e "${BLUE}Theme directory: ${THEME_DIR}${NC}"
echo -e "${BLUE}Hugo root: ${HUGO_ROOT}${NC}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Building Content for Hugo Site ===${NC}"
echo -e "${BLUE}Hugo root: ${HUGO_ROOT}${NC}"

# Create a virtual environment if it doesn't exist
VENV_DIR="${SCRIPT_DIR}/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
else
    echo -e "${YELLOW}Using existing virtual environment...${NC}"
fi

# Activate the virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "${VENV_DIR}/bin/activate"

# Install or upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${YELLOW}Installing requirements...${NC}"
pip install -r "${SCRIPT_DIR}/requirements.txt"

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${YELLOW}Checking for OpenAI API key...${NC}"
    if [ ! -f "${SCRIPT_DIR}/.env" ]; then
        echo -e "${YELLOW}No .env file found. Please enter your OpenAI API key:${NC}"
        read -p "OpenAI API Key: " api_key
        echo "OPENAI_API_KEY=${api_key}" > "${SCRIPT_DIR}/.env"
    fi
fi

# Run the translation script
echo -e "${YELLOW}Running translation script...${NC}"
python "${SCRIPT_DIR}/translate_missing_files.py" --path "${HUGO_ROOT}/content"
echo -e "${GREEN}Translation of missing content completed!${NC}"

# STEP 2: Generate Related Content
echo -e "${BLUE}=== Step 2: Generating Related Content ===${NC}"
python "${SCRIPT_DIR}/generate_related_content.py" --path "${HUGO_ROOT}/content" --hugo-root "${HUGO_ROOT}"

# Deactivate the virtual environment
deactivate

echo -e "${GREEN}Done! All content processing completed successfully.${NC}"
