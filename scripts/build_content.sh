#!/usr/bin/env bash
# build_content.sh
#
# This script creates a virtual environment, installs requirements,
# and performs three main tasks:
# 1. Translates missing content files from English to other languages using FlowHunt API
# 2. Generates related content YAML files for the Hugo site
# 3. Preprocesses images for optimal web delivery

# Re-exec under a newer bash if the resolved one is too old.
# macOS /bin/bash is 3.2 and doesn't support `declare -A` used below.
if [ -z "$BASH_VERSION" ] || [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    for newer in /opt/homebrew/bin/bash /usr/local/bin/bash; do
        if [ -x "$newer" ]; then exec "$newer" "$0" "$@"; fi
    done
    echo "Error: bash 4+ required (script uses associative arrays). Install: brew install bash" >&2
    exit 1
fi

set -e  # Exit immediately if a command exits with a non-zero status

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
THEME_DIR="$(dirname "$SCRIPT_DIR")"
HUGO_ROOT="$(dirname "$(dirname "$THEME_DIR")")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# All available steps
ALL_STEPS=(
    "drop_all_keywords"
    "sync_translations"
    "build_hugo"
    "split_sitemap"
    "offload_images"
    "find_duplicate_images"
    "translate"
    "sync_content_attributes"
    "extract_keywords"
    "sync_translation_urls"
    "generate_translation_urls"
    "generate_amplify_redirects"
    "generate_related_content"
    "generate_clustering"
    "generate_site_audit"
    "generate_paragraph_linkbuilding"
    "preprocess_images"
)

# Step descriptions for the menu
declare -A STEP_DESCRIPTIONS=(
    ["sync_translations"]="Sync translation keys across i18n files"
    ["build_hugo"]="Build Hugo site with minification"
    ["split_sitemap"]="Split sitemap.xml into per-sport sitemaps"
    ["offload_images"]="Offload images from Replicate"
    ["find_duplicate_images"]="Find duplicate images in content"
    ["translate"]="Translate missing content via FlowHunt API"
    ["sync_content_attributes"]="Sync content attributes across languages"
    ["drop_all_keywords"]="DROP ALL keywords (requires regeneration after)"
    ["extract_keywords"]="Extract keywords for files missing them"
    ["sync_translation_urls"]="Sync translation URLs"
    ["generate_translation_urls"]="Generate translation URL mappings"
    ["generate_amplify_redirects"]="Generate AWS Amplify redirects"
    ["generate_related_content"]="Generate related content JSON (split by section)"
    ["generate_clustering"]="Generate website clustering visualization"
    ["generate_site_audit"]="Generate SEO site audit (focus score, radius, outliers, duplicates)"
    ["generate_paragraph_linkbuilding"]="Generate page-local [[lnks]] linkbuilding frontmatter"
    ["preprocess_images"]="Preprocess images for web delivery"
)

install_python_requirements() {
    local requirements_file="$1"
    local torch_requirement="${BUILD_CONTENT_TORCH_REQUIREMENT:-torch>=2.6.0,<2.7}"
    local requirements_without_torch

    echo -e "${YELLOW}Installing Torch dependency: ${torch_requirement}${NC}"
    if ! pip install "$torch_requirement"; then
        if [ -n "${BUILD_CONTENT_TORCH_REQUIREMENT:-}" ]; then
            echo -e "${RED}Failed to install BUILD_CONTENT_TORCH_REQUIREMENT=${BUILD_CONTENT_TORCH_REQUIREMENT}${NC}"
            return 1
        fi

        echo -e "${YELLOW}Torch 2.6.x is not available for this Python/package index; falling back to latest available Torch.${NC}"
        pip install "torch>=2.9.0"

        if [ -z "${FLOWHUNT_EMBEDDING_DEVICE:-}" ]; then
            export FLOWHUNT_EMBEDDING_DEVICE=cpu
            echo -e "${YELLOW}Using FLOWHUNT_EMBEDDING_DEVICE=cpu with fallback Torch to avoid Apple MPS regressions.${NC}"
        fi
    fi

    requirements_without_torch="$(mktemp)"
    grep -Ev '^[[:space:]]*torch([<>=~![:space:]]|$)' "$requirements_file" > "$requirements_without_torch"
    if ! pip install -r "$requirements_without_torch"; then
        rm -f "$requirements_without_torch"
        return 1
    fi
    rm -f "$requirements_without_torch"
}

prefer_cpu_for_newer_torch() {
    if [ -n "${FLOWHUNT_EMBEDDING_DEVICE:-}" ]; then
        return 0
    fi

    if "${VENV_DIR}/bin/python" - <<'PY' >/dev/null 2>&1
from importlib.metadata import version

parts = version("torch").split("+", 1)[0].split(".")
major, minor = int(parts[0]), int(parts[1])
raise SystemExit(0 if (major, minor) >= (2, 7) else 1)
PY
    then
        export FLOWHUNT_EMBEDDING_DEVICE=cpu
        echo -e "${YELLOW}Detected Torch >=2.7; using FLOWHUNT_EMBEDDING_DEVICE=cpu to avoid Apple MPS regressions.${NC}"
    fi
}

# Steps that should be unchecked by default
UNCHECKED_BY_DEFAULT=("offload_images" "find_duplicate_images" "generate_clustering" "generate_site_audit" "generate_paragraph_linkbuilding" "preprocess_images" "drop_all_keywords")

# Interactive checkbox menu function
show_interactive_menu() {
    local num_steps=${#ALL_STEPS[@]}
    declare -a selected

    # Initialize selection state (1=selected, 0=not selected)
    for i in "${!ALL_STEPS[@]}"; do
        local step="${ALL_STEPS[$i]}"
        # Check if step should be unchecked by default
        if [[ " ${UNCHECKED_BY_DEFAULT[*]} " =~ " ${step} " ]]; then
            selected[$i]=0
        else
            selected[$i]=1
        fi
    done

    local current=0
    local key=""

    # Hide cursor
    tput civis 2>/dev/null || true

    # Trap to restore cursor on exit
    trap 'tput cnorm 2>/dev/null || true; tput sgr0 2>/dev/null || true' EXIT

    while true; do
        # Clear screen and move to top
        clear

        echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${BOLD}${BLUE}║           Build Content - Select Steps to Execute                ║${NC}"
        echo -e "${BOLD}${BLUE}╠══════════════════════════════════════════════════════════════════╣${NC}"
        echo -e "${BOLD}${BLUE}║${NC}  Use ${CYAN}↑/↓${NC} to navigate, ${CYAN}Space${NC} to toggle, ${CYAN}Enter${NC} to confirm         ${BOLD}${BLUE}║${NC}"
        echo -e "${BOLD}${BLUE}║${NC}  Press ${CYAN}a${NC} to select all, ${CYAN}n${NC} to select none, ${CYAN}q${NC} to quit            ${BOLD}${BLUE}║${NC}"
        echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
        echo ""

        for i in "${!ALL_STEPS[@]}"; do
            local step="${ALL_STEPS[$i]}"
            local desc="${STEP_DESCRIPTIONS[$step]}"
            local checkbox
            local line_color=""
            local step_display

            if [ "${selected[$i]}" -eq 1 ]; then
                checkbox="${GREEN}[✓]${NC}"
            else
                checkbox="${RED}[ ]${NC}"
            fi

            # Highlight current selection
            if [ "$i" -eq "$current" ]; then
                line_color="${BOLD}${CYAN}"
                step_display="${line_color}► ${checkbox} ${step}${NC}"
            else
                step_display="  ${checkbox} ${step}"
            fi

            # Truncate description if too long
            if [ ${#desc} -gt 40 ]; then
                desc="${desc:0:37}..."
            fi

            printf "  %-45b ${YELLOW}%s${NC}\n" "$step_display" "$desc"
        done

        echo ""
        echo -e "${BLUE}────────────────────────────────────────────────────────────────────${NC}"

        # Count selected
        local count=0
        for s in "${selected[@]}"; do
            ((count += s)) || true
        done
        echo -e "  ${GREEN}${count}${NC} of ${num_steps} steps selected"
        echo ""

        # Read key
        IFS= read -rsn1 key

        # Handle arrow keys (escape sequences)
        if [[ "$key" == $'\x1b' ]]; then
            read -rsn2 -t 0.1 key || true
            case "$key" in
                '[A') # Up arrow
                    ((current > 0)) && ((current--)) || true
                    ;;
                '[B') # Down arrow
                    ((current < num_steps - 1)) && ((current++)) || true
                    ;;
            esac
        else
            case "$key" in
                ' ') # Space - toggle selection
                    if [ "${selected[$current]}" -eq 1 ]; then
                        selected[$current]=0
                    else
                        selected[$current]=1
                    fi
                    ;;
                '') # Enter - confirm
                    break
                    ;;
                'a'|'A') # Select all
                    for i in "${!selected[@]}"; do
                        selected[$i]=1
                    done
                    ;;
                'n'|'N') # Select none
                    for i in "${!selected[@]}"; do
                        selected[$i]=0
                    done
                    ;;
                'q'|'Q') # Quit
                    tput cnorm 2>/dev/null || true
                    echo -e "${YELLOW}Cancelled by user.${NC}"
                    exit 0
                    ;;
            esac
        fi
    done

    # Restore cursor
    tput cnorm 2>/dev/null || true

    # Build the list of selected steps
    MENU_SELECTED_STEPS=()
    for i in "${!ALL_STEPS[@]}"; do
        if [ "${selected[$i]}" -eq 1 ]; then
            MENU_SELECTED_STEPS+=("${ALL_STEPS[$i]}")
        fi
    done

    clear
    echo -e "${GREEN}Selected steps:${NC}"
    for step in "${MENU_SELECTED_STEPS[@]}"; do
        echo -e "  ${GREEN}✓${NC} $step"
    done
    echo ""
}

#print directories
echo -e "${BLUE}=== Directories ===${NC}"
echo -e "${BLUE}Script directory: ${SCRIPT_DIR}${NC}"
echo -e "${BLUE}Theme directory: ${THEME_DIR}${NC}"
echo -e "${BLUE}Hugo root: ${HUGO_ROOT}${NC}"

echo -e "${BLUE}=== Building Content for Hugo Site ===${NC}"
echo -e "${BLUE}Hugo root: ${HUGO_ROOT}${NC}"

# Setup virtual environment
VENV_DIR="${SCRIPT_DIR}/.venv"

# Check if venv exists and has required packages
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    NEED_INSTALL=true
else
    echo -e "${YELLOW}Using existing virtual environment at $VENV_DIR${NC}"
    # Check if key dependencies are installed
    MISSING_DEPS=false
    if ! "${VENV_DIR}/bin/python" -c "import bs4" 2>/dev/null; then
        echo -e "${YELLOW}Missing: beautifulsoup4${NC}"
        MISSING_DEPS=true
    fi
    if ! "${VENV_DIR}/bin/python" -c "from PIL import Image" 2>/dev/null; then
        echo -e "${YELLOW}Missing: Pillow${NC}"
        MISSING_DEPS=true
    fi
    if ! "${VENV_DIR}/bin/python" -c "import frontmatter" 2>/dev/null; then
        echo -e "${YELLOW}Missing: python-frontmatter${NC}"
        MISSING_DEPS=true
    fi
    if ! "${VENV_DIR}/bin/python" -c "import yaml" 2>/dev/null; then
        echo -e "${YELLOW}Missing: pyyaml${NC}"
        MISSING_DEPS=true
    fi
    if ! "${VENV_DIR}/bin/python" -c "import sklearn" 2>/dev/null; then
        echo -e "${YELLOW}Missing: scikit-learn${NC}"
        MISSING_DEPS=true
    fi
    if ! "${VENV_DIR}/bin/python" -c "import boto3" 2>/dev/null; then
        echo -e "${YELLOW}Missing: boto3${NC}"
        MISSING_DEPS=true
    fi
    if [ "$MISSING_DEPS" = true ]; then
        echo -e "${YELLOW}Virtual environment exists but some dependencies are missing${NC}"
        NEED_INSTALL=true
    else
        echo -e "${GREEN}Virtual environment has all required dependencies${NC}"
        NEED_INSTALL=false
    fi
fi

# Activate the virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "${VENV_DIR}/bin/activate"
prefer_cpu_for_newer_torch

# Only install if needed
if [ "$NEED_INSTALL" = true ]; then
    # Install or upgrade pip
    echo -e "${YELLOW}Upgrading pip...${NC}"
    pip install --upgrade pip

    # Install requirements
    echo -e "${YELLOW}Installing requirements...${NC}"
    install_python_requirements "${SCRIPT_DIR}/requirements.txt"

else
    echo -e "${GREEN}Skipping dependency installation - already installed${NC}"
fi

# FlowHunt API key check moved to translate step where it's actually needed

# Parse arguments for step selection
STEPS_TO_RUN=()
SKIP_MENU=false
MAX_PARALLEL_TRANSLATIONS=100  # Default value for parallel translation processes
# Cap for embedding-heavy steps (site_audit, clustering).
# Each background worker loads a sentence-transformer model (~1–2 GB resident),
# so fanning out across all 20+ language dirs at once OOMs the machine.
# Override via env.
MAX_PARALLEL_EMBED="${MAX_PARALLEL_EMBED:-3}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --step|--steps)
            IFS=',' read -ra STEPS_TO_RUN <<< "$2"
            SKIP_MENU=true
            shift 2
            ;;
        --no-menu)
            # Skip menu and run all default steps
            SKIP_MENU=true
            shift
            ;;
        --max-parallel)
            MAX_PARALLEL_TRANSLATIONS="$2"
            shift 2
            ;;
        --help|-h)
            echo -e "${BLUE}Usage: $0 [OPTIONS]${NC}"
            echo ""
            echo "Options:"
            echo "  --step, --steps STEPS      Comma-separated list of steps to run"
            echo "  --no-menu                  Skip interactive menu, run all default steps"
            echo "  --max-parallel NUM         Maximum number of parallel translation processes (default: 10)"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  FLOWHUNT_API_KEY           Required for the 'translate' step (or set in scripts/.env)"
            echo "  MAX_PARALLEL_EMBED         Cap on parallel embedding workers for ML-heavy steps"
            echo "                             (clustering, site_audit). Default: 3"
            echo "                             Each worker loads ~1-2 GB sentence-transformer model — lower"
            echo "                             this on machines with <32 GB RAM to avoid OOM/swap pressure."
            echo ""
            echo "Available steps:"
            for step in "${ALL_STEPS[@]}"; do
                printf "  %-30s %s\n" "$step" "${STEP_DESCRIPTIONS[$step]}"
            done
            echo ""
            echo "Examples:"
            echo "  $0                                           # Show interactive menu"
            echo "  $0 --steps sync_translations,build_hugo"
            echo "  $0 --no-menu                                 # Run all steps without menu"
            echo "  $0 --steps translate --max-parallel 5        # Run translation with 5 parallel processes"
            echo "  $0 --steps generate_paragraph_linkbuilding      # Generate page-local [[lnks]] frontmatter"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Show interactive menu if running in a terminal and no steps specified via args
if [ ${#STEPS_TO_RUN[@]} -eq 0 ] && [ "$SKIP_MENU" = false ] && [ -t 0 ] && [ -t 1 ]; then
    show_interactive_menu
    STEPS_TO_RUN=("${MENU_SELECTED_STEPS[@]}")
fi

# Function to check for duplicate IDs in i18n YAML files
check_duplicate_i18n_ids() {
    echo -e "${BLUE}=== Checking for duplicate IDs in i18n files ===${NC}"

    local has_duplicates=false
    local RED='\033[0;31m'
    local YELLOW='\033[1;33m'
    local GREEN='\033[0;32m'
    local NC='\033[0m'

    # Check both i18n directories
    local i18n_dirs=(
        "${HUGO_ROOT}/i18n"
        "${THEME_DIR}/i18n"
    )

    for i18n_dir in "${i18n_dirs[@]}"; do
        if [ ! -d "$i18n_dir" ]; then
            echo -e "${YELLOW}Warning: i18n directory not found: $i18n_dir${NC}"
            continue
        fi

        echo -e "${YELLOW}Checking: $i18n_dir${NC}"

        # Find all .yaml files in the i18n directory
        while IFS= read -r yaml_file; do
            if [ -f "$yaml_file" ]; then
                local filename=$(basename "$yaml_file")

                # Extract all keys (lines that match the pattern: key: "value")
                # Using awk to find duplicate keys
                local duplicates=$(awk -F':' '
                    /^[a-zA-Z_][a-zA-Z0-9_.]*:/ {
                        # Extract the key (everything before the first colon)
                        key = $1
                        # Trim whitespace
                        gsub(/^[ \t]+|[ \t]+$/, "", key)

                        # Count occurrences
                        if (key in seen) {
                            if (!printed[key]) {
                                duplicates[key] = seen[key]
                                printed[key] = 1
                            }
                            duplicates[key] = duplicates[key] " " NR
                        } else {
                            seen[key] = NR
                        }
                    }
                    END {
                        for (key in duplicates) {
                            print key ":" duplicates[key]
                        }
                    }
                ' "$yaml_file")

                if [ -n "$duplicates" ]; then
                    has_duplicates=true
                    echo -e "${RED}✗ Found duplicate IDs in: $filename${NC}"
                    echo -e "${RED}  File: $yaml_file${NC}"
                    echo -e "${RED}  Duplicate IDs (key: line numbers):${NC}"
                    while IFS= read -r dup_line; do
                        echo -e "${RED}    - $dup_line${NC}"
                    done <<< "$duplicates"
                    echo ""
                fi
            fi
        done < <(find "$i18n_dir" -name "*.yaml" -type f)
    done

    if [ "$has_duplicates" = true ]; then
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}ERROR: Duplicate IDs found in i18n files!${NC}"
        echo -e "${RED}Please fix the duplicate IDs listed above before continuing.${NC}"
        echo -e "${RED}Each ID must be unique within its file.${NC}"
        echo -e "${RED}========================================${NC}"
        return 1
    else
        echo -e "${GREEN}✓ No duplicate IDs found in i18n files${NC}"
        return 0
    fi
}

run_step() {
    step_name="$1"
    echo -e "${YELLOW}[DEBUG] Starting step: $step_name at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    case "$step_name" in
        sync_translations)
            echo -e "${BLUE}=== Step 0: Syncing Translation Keys ===${NC}"

            # Check for duplicate IDs before syncing
            if ! check_duplicate_i18n_ids; then
                echo -e "${RED}Aborting sync_translations due to duplicate IDs${NC}"
                exit 1
            fi

            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/sync_translations.py${NC}"
            python "${SCRIPT_DIR}/sync_translations.py"
            echo -e "${GREEN}Translation key sync completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step sync_translations finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        offload_images)
            echo -e "${BLUE}=== Step 2: Offload Images from Replicate ===${NC}"
            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/offload_replicate_images.py${NC}"
            python "${SCRIPT_DIR}/offload_replicate_images.py"
            echo -e "${GREEN}Offloading images completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step offload_images finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        find_duplicate_images)
            echo -e "${BLUE}=== Step 2.5: Finding Duplicate Images ===${NC}"
            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/find_duplicate_images.py${NC}"
            python "${SCRIPT_DIR}/find_duplicate_images.py"
            echo -e "${GREEN}Duplicate image search completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step find_duplicate_images finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        translate)
            echo -e "${BLUE}=== Step 3: Translating Missing Content with FlowHunt API ===${NC}"
            
            # Check for FlowHunt API key (only needed for translation)
            # Skip in non-interactive environments (like GitHub Actions)
            if [ -z "$FLOWHUNT_API_KEY" ] && [ -t 0 ]; then
                echo -e "${YELLOW}Checking for FlowHunt API key...${NC}"
                if [ ! -f "${SCRIPT_DIR}/.env" ]; then
                    echo -e "${YELLOW}No .env file found. Please enter your FlowHunt API key:${NC}"
                    read -p "FlowHunt API Key: " flow_api_key
                    echo "FLOWHUNT_API_KEY=${flow_api_key}" >> "${SCRIPT_DIR}/.env"
                elif ! grep -q "FLOWHUNT_API_KEY" "${SCRIPT_DIR}/.env"; then
                    echo -e "${YELLOW}FlowHunt API key not found in .env file. Please enter your FlowHunt API key:${NC}"
                    read -p "FlowHunt API Key: " flow_api_key
                    echo "FLOWHUNT_API_KEY=${flow_api_key}" >> "${SCRIPT_DIR}/.env"
                fi
            elif [ -z "$FLOWHUNT_API_KEY" ]; then
                echo -e "${YELLOW}Warning: FlowHunt API key not set. Translation step will be skipped.${NC}"
                echo -e "${YELLOW}To use translation, set FLOWHUNT_API_KEY environment variable.${NC}"
                exit 0
            fi
            
            echo -e "${YELLOW}Running FlowHunt translation script with max ${MAX_PARALLEL_TRANSLATIONS} parallel processes...${NC}"
            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/translate_with_flowhunt.py --path ${HUGO_ROOT}/content --max-scheduled-tasks ${MAX_PARALLEL_TRANSLATIONS}${NC}"
            python "${SCRIPT_DIR}/translate_with_flowhunt.py" --path "${HUGO_ROOT}/content" --max-scheduled-tasks "${MAX_PARALLEL_TRANSLATIONS}"
            echo -e "${GREEN}Translation of missing content completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step translate finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        sync_content_attributes)
            echo -e "${BLUE}=== Step 3.5: Syncing Content Attributes ===${NC}"
            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/sync_content_attributes.py${NC}"
            python "${SCRIPT_DIR}/sync_content_attributes.py"
            echo -e "${GREEN}Content attributes sync completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step sync_content_attributes finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        extract_keywords)
            echo -e "${BLUE}=== Step 3.55: Extracting Keywords for Files Missing Them ===${NC}"
            echo -e "${YELLOW}Running YAKE keyword extraction (only files without keywords)...${NC}"

            # Run once for all content - script handles parallelism internally
            # YAKE is fast, can use more workers
            "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/extract_keywords.py" \
                "${HUGO_ROOT}/content" \
                --recursive \
                --workers 8

            echo -e "${GREEN}Keyword extraction completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step extract_keywords finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        drop_all_keywords)
            echo -e "${BLUE}=== Step: DROP ALL Keywords ===${NC}"
            echo -e "${YELLOW}Dropping keywords from all markdown files using Python...${NC}"
            # Use the Python function from extract_keywords.py for reliable TOML handling
            "${VENV_DIR}/bin/python" -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from pathlib import Path
from extract_keywords import drop_all_keywords

content_path = Path('${HUGO_ROOT}/content')
files = list(content_path.rglob('*.md'))
dropped = drop_all_keywords(files)
print(f'Done. Dropped keywords from {dropped} files.')
"
            echo -e "${GREEN}All keywords dropped! Run extract_keywords to regenerate.${NC}"
            echo -e "${YELLOW}[DEBUG] Step drop_all_keywords finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        sync_translation_urls)
            echo -e "${BLUE}=== Step 3.6: Syncing Translation URLs ===${NC}"
            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/sync_translation_urls.py --hugo-root ${HUGO_ROOT}${NC}"
            python "${SCRIPT_DIR}/sync_translation_urls.py" --hugo-root "${HUGO_ROOT}"
            echo -e "${GREEN}Translation URLs synced!${NC}"
            echo -e "${YELLOW}[DEBUG] Step sync_translation_urls finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        generate_translation_urls)
            echo -e "${BLUE}=== Step 3.7: Generating Translation URLs Mapping ===${NC}"
            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/translation-urls.py --hugo-root ${HUGO_ROOT}${NC}"
            python "${SCRIPT_DIR}/translation-urls.py" --hugo-root "${HUGO_ROOT}"
            echo -e "${GREEN}Translation URLs mapping completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step generate_translation_urls finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        generate_amplify_redirects)
            echo -e "${BLUE}=== Step 3.8: Generating Amplify Redirects ===${NC}"
            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/generate_amplify_redirects_file.py --hugo-root ${HUGO_ROOT}${NC}"
            python "${SCRIPT_DIR}/generate_amplify_redirects_file.py" --hugo-root "${HUGO_ROOT}"
            echo -e "${GREEN}Amplify redirects generation completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step generate_amplify_redirects finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        generate_paragraph_linkbuilding)
            echo -e "${BLUE}=== Step 3.9: Generating Page-local Linkbuilding [[lnks]] ===${NC}"
            echo -e "${YELLOW}Generating paragraph-aware [[lnks]] for all content languages with one model load...${NC}"
            mkdir -p "${HUGO_ROOT}/data/linkbuilding"
            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/generate_paragraph_linkbuilding.py --all-languages --write --remove-old-linkbuilding${NC}"

            "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/generate_paragraph_linkbuilding.py" \
                --all-languages \
                --write \
                --remove-old-linkbuilding

            echo -e "${GREEN}Paragraph linkbuilding generation completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step generate_paragraph_linkbuilding finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        generate_related_content)
            echo -e "${BLUE}=== Step 4: Generating Related Content JSON Files (Split by Section) ===${NC}"
            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/generate_related_content.py --path ${HUGO_ROOT}/content --hugo-root ${HUGO_ROOT} --exclude-sections author${NC}"
            python "${SCRIPT_DIR}/generate_related_content.py" --path "${HUGO_ROOT}/content" --hugo-root "${HUGO_ROOT}" --exclude-sections "author"
            echo -e "${GREEN}Related content JSON generation completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step generate_related_content finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        generate_clustering)
            echo -e "${BLUE}=== Step 4.2: Generating Website Clustering Visualization ===${NC}"
            echo -e "${YELLOW}Running clustering generation for all languages...${NC}"

            # Start clustering in parallel, capped at MAX_PARALLEL_EMBED workers
            # to avoid OOM (each worker loads a ~1–2 GB sentence-transformer model).
            echo -e "${YELLOW}Starting clustering generation (max ${MAX_PARALLEL_EMBED} parallel)...${NC}"
            pids=()
            active=0

            for lang_dir in "${HUGO_ROOT}/content"/*; do
                if [ -d "$lang_dir" ]; then
                    lang=$(basename "$lang_dir")
                    echo -e "${YELLOW}[DEBUG] Starting clustering generation for language: $lang${NC}"

                    # Run clustering in background (with virtual environment)
                    # Set env vars to avoid multiprocessing semaphore leaks
                    (
                        TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
                        "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/generate_clustering.py" \
                            --lang "$lang" \
                            --hugo-root "${HUGO_ROOT}" \
                            --exclude-sections author 2>&1 | \
                            sed "s/^/[$lang] /"

                        if [ ${PIPESTATUS[0]} -eq 0 ]; then
                            echo -e "${GREEN}[$lang] Clustering generated successfully${NC}"
                        else
                            echo -e "${YELLOW}[$lang] Warning: Failed to generate clustering${NC}"
                        fi
                    ) &

                    pids+=($!)
                    active=$((active + 1))
                    if (( active >= MAX_PARALLEL_EMBED )); then
                        wait -n 2>/dev/null || wait "${pids[0]}"
                        active=$((active - 1))
                    fi
                fi
            done

            # Wait for remaining background processes
            echo -e "${YELLOW}Waiting for remaining language clustering to complete...${NC}"
            failed_langs=()
            for pid in "${pids[@]}"; do
                wait $pid
                if [ $? -ne 0 ]; then
                    failed_langs+=("$pid")
                fi
            done

            # Report results
            if [ ${#failed_langs[@]} -eq 0 ]; then
                echo -e "${GREEN}All clustering generations completed successfully!${NC}"
            else
                echo -e "${YELLOW}Some clustering generations failed, but continuing...${NC}"
            fi

            echo -e "${GREEN}Clustering generation completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step generate_clustering finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        generate_site_audit)
            echo -e "${BLUE}=== Step 4.3: Generating SEO Site Audit ===${NC}"
            echo -e "${YELLOW}Computing siteFocusScore, siteRadius, outliers, near-duplicates (max ${MAX_PARALLEL_EMBED} parallel)...${NC}"

            pids=()
            active=0

            for lang_dir in "${HUGO_ROOT}/content"/*; do
                if [ -d "$lang_dir" ]; then
                    lang=$(basename "$lang_dir")
                    echo -e "${YELLOW}[DEBUG] Starting site audit for language: $lang${NC}"

                    (
                        TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
                        "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/generate_site_audit.py" \
                            --lang "$lang" \
                            --hugo-root "${HUGO_ROOT}" 2>&1 | \
                            sed "s/^/[$lang] /"

                        if [ ${PIPESTATUS[0]} -eq 0 ]; then
                            echo -e "${GREEN}[$lang] Site audit generated successfully${NC}"
                        else
                            echo -e "${YELLOW}[$lang] Warning: Failed to generate site audit${NC}"
                        fi
                    ) &

                    pids+=($!)
                    active=$((active + 1))
                    if (( active >= MAX_PARALLEL_EMBED )); then
                        wait -n 2>/dev/null || wait "${pids[0]}"
                        active=$((active - 1))
                    fi
                fi
            done

            echo -e "${YELLOW}Waiting for remaining language site audits to complete...${NC}"
            failed_langs=()
            for pid in "${pids[@]}"; do
                wait $pid
                if [ $? -ne 0 ]; then
                    failed_langs+=("$pid")
                fi
            done

            if [ ${#failed_langs[@]} -eq 0 ]; then
                echo -e "${GREEN}All site audits completed successfully!${NC}"
            else
                echo -e "${YELLOW}Some site audits failed, but continuing...${NC}"
            fi

            echo -e "${GREEN}Site audit generation completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step generate_site_audit finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        build_hugo)
            echo -e "${BLUE}=== Step 2: Building Hugo Site (All Languages) ===${NC}"
            echo -e "${YELLOW}Building Hugo site to validate content and generate HTML files...${NC}"
            
            # Build Hugo with minification (all languages)
            echo -e "${YELLOW}[DEBUG] Executing: hugo --minify --buildFuture${NC}"
            
            cd "${HUGO_ROOT}"
            hugo --minify --buildFuture
            
            if [ $? -ne 0 ]; then
                echo -e "${RED}ERROR: Hugo build failed! Content has errors that must be fixed.${NC}"
                echo -e "${RED}Please fix the errors above before continuing.${NC}"
                exit 1
            fi
            
            echo -e "${GREEN}Hugo build completed successfully!${NC}"
            
            # Count generated files per language
            total_count=$(find "${HUGO_ROOT}/public" -name "*.html" | wc -l)
            echo -e "${GREEN}Generated ${total_count} total HTML files${NC}"
            
            # Show breakdown by language
            echo -e "${YELLOW}Files per language:${NC}"
            for lang_dir in ar cs da de es fi fr it ja ko nl no pl pt ro sk sv tr vi zh; do
                if [ -d "${HUGO_ROOT}/public/${lang_dir}" ]; then
                    lang_count=$(find "${HUGO_ROOT}/public/${lang_dir}" -name "*.html" | wc -l)
                    echo -e "  ${lang_dir}: ${lang_count} files"
                fi
            done
            # English is at root
            en_count=$(find "${HUGO_ROOT}/public" -maxdepth 3 -name "*.html" -not -path "*/ar/*" -not -path "*/cs/*" -not -path "*/da/*" -not -path "*/de/*" -not -path "*/es/*" -not -path "*/fi/*" -not -path "*/fr/*" -not -path "*/it/*" -not -path "*/ja/*" -not -path "*/ko/*" -not -path "*/nl/*" -not -path "*/no/*" -not -path "*/pl/*" -not -path "*/pt/*" -not -path "*/ro/*" -not -path "*/sk/*" -not -path "*/sv/*" -not -path "*/tr/*" -not -path "*/vi/*" -not -path "*/zh/*" | wc -l)
            echo -e "  en: ${en_count} files (at root)"
            
            echo -e "${YELLOW}[DEBUG] Step build_hugo finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        split_sitemap)
            echo -e "${BLUE}=== Step 2.1: Splitting Sitemap ===${NC}"
            echo -e "${YELLOW}Splitting monolithic sitemap.xml into per-sport sitemaps...${NC}"
            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/split_sitemap.py --public-dir ${HUGO_ROOT}/public${NC}"
            python "${SCRIPT_DIR}/split_sitemap.py" --public-dir "${HUGO_ROOT}/public"
            echo -e "${GREEN}Sitemap splitting completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step split_sitemap finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        preprocess_images)
            echo -e "${BLUE}=== Step 5: Preprocessing Images ===${NC}"
            echo -e "${YELLOW}Running image preprocessing script...${NC}"
            echo -e "${YELLOW}[DEBUG] Executing: source ${SCRIPT_DIR}/preprocess-images.sh && process_all_images${NC}"
            source "${SCRIPT_DIR}/preprocess-images.sh"
            process_all_images
            echo -e "${GREEN}Image preprocessing completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step preprocess_images finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        *)
            echo -e "${YELLOW}Unknown step: $step_name${NC}"
            ;;
    esac
}

# If no steps specified (non-interactive mode), run default steps (excluding find_duplicate_images)
if [ ${#STEPS_TO_RUN[@]} -eq 0 ]; then
    # In non-interactive mode, run all steps except those in UNCHECKED_BY_DEFAULT
    for step in "${ALL_STEPS[@]}"; do
        if [[ ! " ${UNCHECKED_BY_DEFAULT[*]} " =~ " ${step} " ]]; then
            STEPS_TO_RUN+=("$step")
        fi
    done
    echo -e "${YELLOW}[DEBUG] No steps specified, running default steps: ${STEPS_TO_RUN[@]}${NC}"
else
    echo -e "${YELLOW}[DEBUG] Running specified steps: ${STEPS_TO_RUN[@]}${NC}"
fi

# Show confirmation of steps to run
echo -e "${BLUE}=== Steps to execute ===${NC}"
for step in "${STEPS_TO_RUN[@]}"; do
    echo -e "  ${GREEN}✓${NC} $step - ${STEP_DESCRIPTIONS[$step]}"
done
echo ""

echo -e "${YELLOW}[DEBUG] Starting main processing loop at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
for step in "${STEPS_TO_RUN[@]}"; do
    echo -e "${YELLOW}[DEBUG] Preparing to run step: $step${NC}"
    run_step "$step"
    echo -e "${YELLOW}[DEBUG] Completed step: $step${NC}"
done
echo -e "${YELLOW}[DEBUG] Main processing loop finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"

# Deactivate the virtual environment
deactivate

echo -e "${GREEN}Done! All content processing completed successfully.${NC}"
