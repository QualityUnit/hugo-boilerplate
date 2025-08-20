#!/bin/bash

# Simple preview server using Python's built-in HTTP server

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PORT=${1:-8181}

# Find the project root (where public directory should be)
PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
PUBLIC_DIR="$PROJECT_ROOT/public"

# Check if public directory exists
if [ ! -d "$PUBLIC_DIR" ]; then
    echo -e "${RED}Error: public/ directory not found at $PUBLIC_DIR${NC}"
    echo "Please run 'hugo' to generate the site first."
    exit 1
fi

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    # Check if it's Python 3
    if python --version 2>&1 | grep -q "Python 3"; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}Python 3 is required but not found${NC}"
        exit 1
    fi
else
    echo -e "${RED}Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   Hugo Preview Server${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Starting simple HTTP server on port ${PORT}...${NC}"
echo -e "${BLUE}📁 Serving from: $PUBLIC_DIR${NC}"
echo -e "${GREEN}📎 Access the site at: http://localhost:${PORT}${NC}"
echo ""
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Start the server
cd "$PUBLIC_DIR"
exec "$PYTHON_CMD" -m http.server "$PORT" --bind 127.0.0.1