#!/bin/bash
# build_content.sh
#
# This script creates a virtual environment, installs requirements,
# and performs three main tasks:
# 1. Translates missing content files from English to other languages using FlowHunt API
# 2. Generates related content YAML files for the Hugo site
# 3. Preprocesses images for optimal web delivery

set -e  # Exit immediately if a command exits with a non-zero status

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
THEME_DIR="$(dirname "$SCRIPT_DIR")"
HUGO_ROOT="$(dirname "$(dirname "$THEME_DIR")")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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
    # Check if beautifulsoup4 is installed (as a proxy for all deps)
    if ! "${VENV_DIR}/bin/python" -c "import bs4" 2>/dev/null; then
        echo -e "${YELLOW}Virtual environment exists but dependencies are missing${NC}"
        NEED_INSTALL=true
    else
        echo -e "${GREEN}Virtual environment has required dependencies${NC}"
        NEED_INSTALL=false
    fi
fi

# Activate the virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "${VENV_DIR}/bin/activate"

# Only install if needed
if [ "$NEED_INSTALL" = true ]; then
    # Install or upgrade pip
    echo -e "${YELLOW}Upgrading pip...${NC}"
    pip install --upgrade pip
    
    # Install requirements
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -r "${SCRIPT_DIR}/requirements.txt"
else
    echo -e "${GREEN}Skipping dependency installation - already installed${NC}"
fi

# FlowHunt API key check moved to translate step where it's actually needed

# Parse arguments for step selection
STEPS_TO_RUN=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --step|--steps)
            IFS=',' read -ra STEPS_TO_RUN <<< "$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

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
            
            echo -e "${YELLOW}Running FlowHunt translation script...${NC}"
            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/translate_with_flowhunt.py --path ${HUGO_ROOT}/content${NC}"
            python "${SCRIPT_DIR}/translate_with_flowhunt.py" --path "${HUGO_ROOT}/content"
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
        generate_linkbuilding_keywords)
            echo -e "${BLUE}=== Step 3.9: Generating Linkbuilding Keywords ===${NC}"
            echo -e "${YELLOW}Running linkbuilding keyword generation for all languages...${NC}"
            
            # Start all language extractions in parallel
            echo -e "${YELLOW}Starting parallel linkbuilding keyword generation for all languages...${NC}"
            pids=()
            
            for lang_dir in "${HUGO_ROOT}/content"/*; do
                if [ -d "$lang_dir" ]; then
                    lang=$(basename "$lang_dir")
                    echo -e "${YELLOW}[DEBUG] Starting linkbuilding keyword generation for language: $lang${NC}"
                    
                    # Run generation in background (with virtual environment)
                    (
                        "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/generate_linkbuilding_keywords.py" \
                            --lang "$lang" \
                            --top-k 10 \
                            --min-keyword-freq 3 \
                            --min-files 5 \
                            --min-ngram 2 \
                            --max-ngram 4 2>&1 | \
                            sed "s/^/[$lang] /"
                        
                        if [ ${PIPESTATUS[0]} -eq 0 ]; then
                            echo -e "${GREEN}[$lang] Linkbuilding keywords generated successfully${NC}"
                        else
                            echo -e "${YELLOW}[$lang] Warning: Failed to generate linkbuilding keywords${NC}"
                        fi
                    ) &
                    
                    # Store the PID
                    pids+=($!)
                fi
            done
            
            # Wait for all background processes to complete
            echo -e "${YELLOW}Waiting for all language linkbuilding generations to complete...${NC}"
            failed_langs=()
            for pid in "${pids[@]}"; do
                wait $pid
                if [ $? -ne 0 ]; then
                    failed_langs+=("$pid")
                fi
            done
            
            # Report results
            if [ ${#failed_langs[@]} -eq 0 ]; then
                echo -e "${GREEN}All linkbuilding keyword generations completed successfully!${NC}"
            else
                echo -e "${YELLOW}Some linkbuilding generations failed, but continuing...${NC}"
            fi
            
            echo -e "${GREEN}Linkbuilding keyword generation completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step generate_linkbuilding_keywords finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        generate_related_content)
            echo -e "${BLUE}=== Step 4: Generating Related Content ===${NC}"
            echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/generate_related_content.py --path ${HUGO_ROOT}/content --hugo-root ${HUGO_ROOT} --exclude-sections author${NC}"
            python "${SCRIPT_DIR}/generate_related_content.py" --path "${HUGO_ROOT}/content" --hugo-root "${HUGO_ROOT}" --exclude-sections "author"
            echo -e "${GREEN}Related content generation completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step generate_related_content finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        extract_automatic_links)
            echo -e "${BLUE}=== Step 4.5: Extracting Automatic Links ===${NC}"
            echo -e "${YELLOW}Extracting keywords from markdown frontmatter for linkbuilding...${NC}"
            
            # Create linkbuilding directory if it doesn't exist
            mkdir -p "${HUGO_ROOT}/data/linkbuilding"
            
            # Start all language extractions in parallel
            echo -e "${YELLOW}Starting parallel extraction for all languages...${NC}"
            pids=()
            
            for lang_dir in "${HUGO_ROOT}/content"/*; do
                if [ -d "$lang_dir" ]; then
                    lang=$(basename "$lang_dir")
                    echo -e "${YELLOW}[DEBUG] Starting extraction for language: $lang${NC}"
                    
                    # Run extraction in background (with virtual environment)
                    (
                        "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/extract_automatic_links.py" \
                            --content-dir "${lang_dir}/" \
                            --output "${HUGO_ROOT}/data/linkbuilding/${lang}_automatic.json" 2>&1 | \
                            sed "s/^/[$lang] /"
                        
                        if [ ${PIPESTATUS[0]} -eq 0 ]; then
                            echo -e "${GREEN}[$lang] Automatic links extracted successfully${NC}"
                        else
                            echo -e "${YELLOW}[$lang] Warning: Failed to extract automatic links${NC}"
                        fi
                    ) &
                    
                    # Store the PID
                    pids+=($!)
                fi
            done
            
            # Wait for all background processes to complete
            echo -e "${YELLOW}Waiting for all language extractions to complete...${NC}"
            failed_langs=()
            for pid in "${pids[@]}"; do
                wait $pid
                if [ $? -ne 0 ]; then
                    failed_langs+=("$pid")
                fi
            done
            
            # Report results
            if [ ${#failed_langs[@]} -eq 0 ]; then
                echo -e "${GREEN}All automatic link extractions completed successfully!${NC}"
            else
                echo -e "${YELLOW}Some extractions failed, but continuing...${NC}"
            fi
            
            echo -e "${GREEN}Automatic link extraction completed!${NC}"
            echo -e "${YELLOW}[DEBUG] Step extract_automatic_links finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
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
        precompute_linkbuilding)
            echo -e "${BLUE}=== Step 4.8: Precomputing Optimized Linkbuilding ===${NC}"
            echo -e "${YELLOW}Analyzing content to optimize linkbuilding keywords...${NC}"
            
            # Ensure Hugo has been built
            if [ ! -d "${HUGO_ROOT}/public" ]; then
                echo -e "${YELLOW}Warning: Public directory not found. Building Hugo first...${NC}"
                echo -e "${YELLOW}[DEBUG] Executing: hugo --minify --buildFuture${NC}"
                cd "${HUGO_ROOT}"
                hugo --minify --buildFuture
                if [ $? -ne 0 ]; then
                    echo -e "${YELLOW}Error: Hugo build failed. Cannot proceed with linkbuilding optimization.${NC}"
                    exit 1
                fi
            else
                # Check if public directory is recent
                if [ -f "${HUGO_ROOT}/public/index.html" ]; then
                    public_age=$(( $(date +%s) - $(stat -f%m "${HUGO_ROOT}/public/index.html" 2>/dev/null || stat -c%Y "${HUGO_ROOT}/public/index.html" 2>/dev/null) ))
                    if [ $public_age -gt 3600 ]; then
                        echo -e "${YELLOW}Public directory is over 1 hour old. Rebuilding Hugo (all languages)...${NC}"
                        cd "${HUGO_ROOT}"
                        hugo --minify --buildFuture
                    else
                        echo -e "${GREEN}Public directory is recent (less than 1 hour old), skipping rebuild${NC}"
                    fi
                fi
            fi
            
            if [ -d "${HUGO_ROOT}/public" ]; then
                # Create optimized directory
                mkdir -p "${HUGO_ROOT}/data/linkbuilding/optimized"
                
                echo -e "${YELLOW}[DEBUG] Executing: python ${SCRIPT_DIR}/precompute_linkbuilding.py${NC}"
                
                "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/precompute_linkbuilding.py" \
                    --linkbuilding-dir "${HUGO_ROOT}/data/linkbuilding" \
                    --public-dir "${HUGO_ROOT}/public" \
                    --output-dir "${HUGO_ROOT}/data/linkbuilding/optimized" \
                    --max-workers 4
                
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}Linkbuilding optimization completed!${NC}"
                    
                    # Show summary
                    if [ -f "${HUGO_ROOT}/data/linkbuilding/optimized/precomputation_summary.json" ]; then
                        echo -e "${YELLOW}Optimization summary:${NC}"
                        "${VENV_DIR}/bin/python" -c "
import json
with open('${HUGO_ROOT}/data/linkbuilding/optimized/precomputation_summary.json', 'r') as f:
    data = json.load(f)
    summary = data.get('summary', {})
    print(f'  Original keywords: {summary.get(\"total_original_keywords\", 0):,}')
    print(f'  Keywords found in content: {summary.get(\"total_found_keywords\", 0):,}')
    print(f'  Average reduction: {summary.get(\"average_reduction_percent\", 0):.1f}%')
"
                    fi
                else
                    echo -e "${YELLOW}Warning: Linkbuilding optimization failed${NC}"
                fi
            fi
            
            echo -e "${YELLOW}[DEBUG] Step precompute_linkbuilding finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
            ;;
        apply_linkbuilding)
            echo -e "${BLUE}=== Step 4.6: Applying Linkbuilding (Parallel) ===${NC}"
            echo -e "${YELLOW}Running parallel linkbuilding for all languages...${NC}"
            
            # Log environment details for debugging
            echo -e "${YELLOW}[DEBUG] Environment details:${NC}"
            echo -e "  - Current directory: $(pwd)"
            echo -e "  - HUGO_ROOT: ${HUGO_ROOT}"
            echo -e "  - VENV_DIR: ${VENV_DIR}"
            echo -e "  - VENV_DIR exists: $([ -d "${VENV_DIR}" ] && echo 'YES' || echo 'NO')"
            echo -e "  - Python binary exists: $([ -f "${VENV_DIR}/bin/python" ] && echo 'YES' || echo 'NO')"
            
            # Check if the venv Python is accessible
            if [ -f "${VENV_DIR}/bin/python" ]; then
                echo -e "  - Python version: $(${VENV_DIR}/bin/python --version 2>&1)"
                
                # Check Python dependencies
                echo -e "${YELLOW}[DEBUG] Checking Python dependencies:${NC}"
                ${VENV_DIR}/bin/python -c "import bs4; print('  ✓ beautifulsoup4 version:', bs4.__version__)" 2>&1 || echo -e "  ${RED}✗ beautifulsoup4 NOT FOUND${NC}"
                ${VENV_DIR}/bin/python -c "import yaml; print('  ✓ pyyaml installed')" 2>&1 || echo -e "  ${RED}✗ pyyaml NOT FOUND${NC}"
            else
                echo -e "  ${RED}ERROR: Python binary not found at ${VENV_DIR}/bin/python${NC}"
                echo -e "  ${RED}Virtual environment may not be properly initialized${NC}"
                exit 1
            fi
            
            # Check if public directory exists
            if [ ! -d "${HUGO_ROOT}/public" ]; then
                echo -e "${RED}ERROR: Public directory not found at ${HUGO_ROOT}/public${NC}"
                echo -e "${RED}Cannot run linkbuilding without Hugo build output.${NC}"
                echo -e "${YELLOW}Please ensure Hugo build completes successfully before linkbuilding.${NC}"
                exit 1
            else
                # Check what language directories were actually built
                echo -e "${YELLOW}Checking built language directories:${NC}"
                for lang_code in en ar cs da de es fi fr it ja ko nl no pl pt ro sk sv tr vi zh; do
                    if [ "$lang_code" = "en" ]; then
                        # English is at root
                        if [ -f "${HUGO_ROOT}/public/index.html" ]; then
                            echo -e "  ✓ English (en) - found at root"
                        else
                            echo -e "  ✗ English (en) - NOT FOUND"
                        fi
                    else
                        if [ -d "${HUGO_ROOT}/public/${lang_code}" ]; then
                            file_count=$(find "${HUGO_ROOT}/public/${lang_code}" -name "*.html" 2>/dev/null | wc -l)
                            echo -e "  ✓ ${lang_code} - ${file_count} HTML files"
                        else
                            echo -e "  ✗ ${lang_code} - directory not found"
                        fi
                    fi
                done
                
                # Check linkbuilding data directory
                echo -e "${YELLOW}[DEBUG] Checking linkbuilding data:${NC}"
                if [ -d "${HUGO_ROOT}/data/linkbuilding" ]; then
                    echo -e "  ✓ Linkbuilding data directory exists"
                    file_count=$(find "${HUGO_ROOT}/data/linkbuilding" -name "*.json" -o -name "*.yaml" 2>/dev/null | wc -l)
                    echo -e "  - Found ${file_count} data files"
                else
                    echo -e "  ${RED}✗ Linkbuilding data directory NOT FOUND at ${HUGO_ROOT}/data/linkbuilding${NC}"
                fi
                
                echo -e "${YELLOW}[DEBUG] Executing linkbuilding command:${NC}"
                echo -e "  ${VENV_DIR}/bin/python ${SCRIPT_DIR}/linkbuilding_parallel.py \\"
                echo -e "    --linkbuilding-dir ${HUGO_ROOT}/data/linkbuilding \\"
                echo -e "    --public-dir ${HUGO_ROOT}/public \\"
                echo -e "    --script-path ${SCRIPT_DIR}/linkbuilding.py \\"
                echo -e "    --max-workers 8"
                
                # Run with explicit error capture
                ERROR_LOG=$(mktemp)
                "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/linkbuilding_parallel.py" \
                    --linkbuilding-dir "${HUGO_ROOT}/data/linkbuilding" \
                    --public-dir "${HUGO_ROOT}/public" \
                    --script-path "${SCRIPT_DIR}/linkbuilding.py" \
                    --max-workers 8 2>"${ERROR_LOG}"
                
                EXIT_CODE=$?
                
                # Show any error output
                if [ -s "${ERROR_LOG}" ]; then
                    echo -e "${YELLOW}[DEBUG] Stderr output from linkbuilding:${NC}"
                    cat "${ERROR_LOG}"
                fi
                rm -f "${ERROR_LOG}"
                
                if [ $EXIT_CODE -eq 0 ]; then
                    echo -e "${GREEN}Parallel linkbuilding completed successfully!${NC}"
                else
                    echo -e "${RED}ERROR: Linkbuilding failed with exit code $EXIT_CODE${NC}"
                    echo -e "${YELLOW}Please check the error messages above for details.${NC}"
                    # Exit with error code to properly report failure
                    exit $EXIT_CODE
                fi
            fi
            
            echo -e "${YELLOW}[DEBUG] Step apply_linkbuilding finished at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
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

# If no steps specified, run all steps
if [ ${#STEPS_TO_RUN[@]} -eq 0 ]; then
    STEPS_TO_RUN=(sync_translations build_hugo offload_images find_duplicate_images translate sync_content_attributes sync_translation_urls generate_translation_urls generate_amplify_redirects generate_related_content generate_linkbuilding_keywords extract_automatic_links precompute_linkbuilding preprocess_images)
    echo -e "${YELLOW}[DEBUG] No steps specified, running all steps: ${STEPS_TO_RUN[@]}${NC}"
else
    echo -e "${YELLOW}[DEBUG] Running specified steps: ${STEPS_TO_RUN[@]}${NC}"
fi

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
