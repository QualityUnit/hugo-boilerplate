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

# Check for FlowHunt API key
if [ -z "$FLOWHUNT_API_KEY" ]; then
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
fi

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

run_step() {
    step_name="$1"
    echo -e "${YELLOW}[DEBUG] Starting step: $step_name at $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    case "$step_name" in
        sync_translations)
            echo -e "${BLUE}=== Step 0: Syncing Translation Keys ===${NC}"
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
            echo -e "${YELLOW}[DEBUG] Executing: hugo --minify${NC}"
            
            cd "${HUGO_ROOT}"
            hugo --minify
            
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
                echo -e "${YELLOW}[DEBUG] Executing: hugo --minify${NC}"
                cd "${HUGO_ROOT}"
                hugo --minify
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
                        hugo --minify
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
                    --max-workers 8
                
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
            
            # Check if public directory exists
            if [ ! -d "${HUGO_ROOT}/public" ]; then
                echo -e "${YELLOW}Warning: Public directory not found. Skipping linkbuilding.${NC}"
                echo -e "${YELLOW}Run 'hugo' to generate the public directory first.${NC}"
            else
                echo -e "${YELLOW}[DEBUG] Executing: ${VENV_DIR}/bin/python ${SCRIPT_DIR}/linkbuilding_parallel.py --linkbuilding-dir ${HUGO_ROOT}/data/linkbuilding --public-dir ${HUGO_ROOT}/public${NC}"
                
                "${VENV_DIR}/bin/python" "${SCRIPT_DIR}/linkbuilding_parallel.py" \
                    --linkbuilding-dir "${HUGO_ROOT}/data/linkbuilding" \
                    --public-dir "${HUGO_ROOT}/public" \
                    --script-path "${SCRIPT_DIR}/linkbuilding.py" \
                    --max-workers 8
                
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}Parallel linkbuilding completed successfully!${NC}"
                else
                    echo -e "${YELLOW}Warning: Some languages failed during linkbuilding${NC}"
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
    STEPS_TO_RUN=(sync_translations build_hugo offload_images find_duplicate_images translate sync_content_attributes sync_translation_urls generate_translation_urls generate_related_content extract_automatic_links precompute_linkbuilding preprocess_images)
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
