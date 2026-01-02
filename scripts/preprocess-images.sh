#!/bin/bash

# Image preprocessing script for Wachman
# This script preprocesses images and stores them in the cdn-assets/processed directory
# Images are only processed if they are larger than the size limits
# For WebP images, only size alternatives are created, no format conversion
# If the processed image is larger than the original, it is not used
#
# Performance optimization: Uses a cache file to remember which images have been
# processed and which resulted in larger files (and were thus skipped). This avoids
# re-processing images that won't benefit from optimization.

# Set error handling
set -e

# Get the root directory of the Hugo site
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
THEME_DIR="$(dirname "$SCRIPT_DIR")"
HUGO_ROOT="$(dirname "$(dirname "$THEME_DIR")")"

SOURCE_DIR="$HUGO_ROOT/cdn-assets"
TARGET_DIR="$HUGO_ROOT/cdn-assets/processed"
CACHE_FILE="$HUGO_ROOT/data/image_processing_cache.json"

# Create the processed directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Initialize cache file if it doesn't exist
if [ ! -f "$CACHE_FILE" ]; then
    echo '{}' > "$CACHE_FILE"
fi

# Declare associative arrays for cache (bash 4+)
declare -A CACHE_MTIME
declare -A CACHE_SKIPPED

# Load cache into memory
load_cache() {
    if [ -f "$CACHE_FILE" ]; then
        # Parse JSON cache file into associative arrays
        while IFS= read -r line; do
            # Extract key and values from JSON-like format
            # Format: "relative/path/to/image.jpg": {"mtime": 1234567890, "skipped_variants": ["300", "1024", "webp"]}
            if [[ "$line" =~ \"([^\"]+)\":\ *\{\"mtime\":\ *([0-9]+),\ *\"skipped_variants\":\ *\[([^\]]*)\]\} ]]; then
                local key="${BASH_REMATCH[1]}"
                local mtime="${BASH_REMATCH[2]}"
                local skipped="${BASH_REMATCH[3]}"
                CACHE_MTIME["$key"]="$mtime"
                CACHE_SKIPPED["$key"]="$skipped"
            fi
        done < "$CACHE_FILE"
    fi
}

# Save cache to file
save_cache() {
    local tmp_file="${CACHE_FILE}.tmp"
    echo '{' > "$tmp_file"
    local first=true
    for key in "${!CACHE_MTIME[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo ',' >> "$tmp_file"
        fi
        printf '  "%s": {"mtime": %s, "skipped_variants": [%s]}' \
            "$key" "${CACHE_MTIME[$key]}" "${CACHE_SKIPPED[$key]}" >> "$tmp_file"
    done
    echo '' >> "$tmp_file"
    echo '}' >> "$tmp_file"
    mv "$tmp_file" "$CACHE_FILE"
}

# Get file modification time in seconds since epoch
get_mtime() {
    stat -f%m "$1" 2>/dev/null || stat -c%Y "$1" 2>/dev/null || echo "0"
}

# Check if a variant was skipped (processed but resulted in larger file)
is_variant_skipped() {
    local rel_path="$1"
    local variant="$2"
    local skipped="${CACHE_SKIPPED[$rel_path]}"
    if [[ "$skipped" == *"\"$variant\""* ]]; then
        return 0 # true - variant was skipped
    fi
    return 1 # false - variant was not skipped
}

# Add a variant to the skipped list
add_skipped_variant() {
    local rel_path="$1"
    local variant="$2"
    local current="${CACHE_SKIPPED[$rel_path]}"
    if [ -z "$current" ]; then
        CACHE_SKIPPED["$rel_path"]="\"$variant\""
    elif [[ "$current" != *"\"$variant\""* ]]; then
        CACHE_SKIPPED["$rel_path"]="$current, \"$variant\""
    fi
}

# Update cache entry for an image
update_cache_entry() {
    local rel_path="$1"
    local mtime="$2"
    CACHE_MTIME["$rel_path"]="$mtime"
    # Initialize skipped_variants if not set
    if [ -z "${CACHE_SKIPPED[$rel_path]}" ]; then
        CACHE_SKIPPED["$rel_path"]=""
    fi
}

# Check if image needs processing based on cache
needs_processing_cached() {
    local rel_path="$1"
    local current_mtime="$2"
    local cached_mtime="${CACHE_MTIME[$rel_path]}"

    # If not in cache, needs processing
    if [ -z "$cached_mtime" ]; then
        return 0 # true
    fi

    # If source file has changed, needs processing
    if [ "$current_mtime" != "$cached_mtime" ]; then
        # Clear skipped variants since file changed
        CACHE_SKIPPED["$rel_path"]=""
        return 0 # true
    fi

    return 1 # false - already processed and unchanged
}

# Define image widths - can be adjusted or extended without changing the code logic
# Each width will create a resized version of the image
IMAGE_WIDTHS=(300 1024)

# Quality settings
QUALITY_JPG=95
QUALITY_WEBP=95

# Function to check if the file exists and is newer than the target
needs_processing() {
    local source="$1"
    local target="$2"
    
    # If target doesn't exist, we need to process
    if [ ! -f "$target" ]; then
        return 0 # true
    fi
    
    # If source is newer than target, we need to process
    if [ "$source" -nt "$target" ]; then
        return 0 # true
    fi
    
    return 1 # false
}

# Function to get image dimensions using identify from ImageMagick
get_image_width() {
    magick identify -format "%w" "$1" 2>/dev/null || echo "0"
}

# Function to get file size in bytes
get_file_size() {
    stat -f%z "$1" 2>/dev/null || echo "0"
}

# Function to process an image
process_image() {
    local source="$1"
    local rel_path="${source#$SOURCE_DIR/}"
    local rel_dir=$(dirname "$rel_path")
    local filename=$(basename "$source")
    local extension="${filename##*.}"
    local basename="${filename%.*}"
    local is_webp=false
    local current_mtime=$(get_mtime "$source")
    local did_any_work=false

    # Check if image was already fully processed and unchanged
    if ! needs_processing_cached "$rel_path" "$current_mtime"; then
        # Image is in cache with same mtime - check if all variants exist or are skipped
        local all_done=true

        # Check optimized original
        local optimized_original="$TARGET_DIR/$rel_dir/${basename}.${extension}"
        if [ ! -f "$optimized_original" ] && ! is_variant_skipped "$rel_path" "original"; then
            all_done=false
        fi

        if [ "$all_done" = true ]; then
            echo "Skipping (cached): $rel_path"
            return
        fi
    fi

    echo "Processing: $rel_path"

    # Update cache entry with current mtime
    update_cache_entry "$rel_path" "$current_mtime"

    # Create the target directory structure that mirrors the source directory
    local target_dir="$TARGET_DIR/$rel_dir"
    mkdir -p "$target_dir"

    # Check if it's a WebP image
    if [[ "$extension" == "webp" ]]; then
        is_webp=true
    fi

    # Get original image width and size
    local original_width=$(get_image_width "$source")
    local original_size=$(get_file_size "$source")

    echo "  Original width: $original_width px, size: $original_size bytes"

    # Store optimized original image if it doesn't exist or needs updating
    local optimized_original="$target_dir/${basename}.${extension}"
    if ! is_variant_skipped "$rel_path" "original"; then
        if needs_processing "$source" "$optimized_original"; then
            echo "  Creating optimized original in $target_dir"
            did_any_work=true

            if [ "$is_webp" = true ]; then
                # For WebP, optimize without changing dimensions
                magick "$source" -quality 99 "$optimized_original"
            else
                # For other formats, optimize without changing dimensions
                magick "$source" -quality 99 "$optimized_original"
            fi

            # Check if the optimized image is larger than the original
            local optimized_size=$(get_file_size "$optimized_original")
            echo "    Optimized original size: $optimized_size bytes"
            if [ "$optimized_size" -gt "$original_size" ]; then
                echo "  Warning: Optimized original is larger than original, using original instead"
                cp "$source" "$optimized_original"
                local copied_size=$(get_file_size "$optimized_original")
                echo "    Copied original size: $copied_size bytes"
            fi
        fi
    else
        echo "  Skipping optimized original (cached as no benefit)"
    fi

    # Process each width for the image
    for width in "${IMAGE_WIDTHS[@]}"; do
        # Only process if original is larger than target width
        if [ "$original_width" -gt "$width" ]; then
            local target_file="$target_dir/${basename}-${width}.${extension}"
            local variant_key="${width}"

            # Skip if this variant was previously processed but resulted in larger file
            if is_variant_skipped "$rel_path" "$variant_key"; then
                echo "  Skipping ${width}px (cached as larger than original)"
                continue
            fi

            if needs_processing "$source" "$target_file"; then
                echo "  Creating ${width}px width"
                did_any_work=true

                if [ "$is_webp" = true ]; then
                    # For WebP, resize and set quality
                    magick "$source" -resize "${width}x>" -quality 90 "$target_file"
                else
                    # For other formats, resize and set quality
                    magick "$source" -resize "${width}x>" -quality 90 "$target_file"
                fi

                # Check if the processed image is larger than the original
                local processed_size=$(get_file_size "$target_file")
                echo "    ${width}px version size: $processed_size bytes"
                if [ "$processed_size" -gt "$original_size" ]; then
                    echo "  Warning: ${width}px version is larger than original, removing and caching"
                    rm "$target_file"
                    add_skipped_variant "$rel_path" "$variant_key"
                fi
            fi
        fi
    done

    # For non-WebP images, create a WebP version of the original
    if [ "$is_webp" = false ]; then
        local webp_target="$target_dir/${basename}.webp"

        # Skip if this variant was previously processed but resulted in larger file
        if ! is_variant_skipped "$rel_path" "webp"; then
            if needs_processing "$source" "$webp_target"; then
                echo "  Creating WebP version"
                did_any_work=true
                magick "$source" -quality "$QUALITY_WEBP" "$webp_target"

                # Log the WebP file size but keep it even if larger
                local processed_size=$(get_file_size "$webp_target")
                echo "    WebP version size: $processed_size bytes"
                if [ "$processed_size" -gt "$original_size" ]; then
                    echo "  Note: WebP version is larger than original, but keeping it for browser compatibility"
                fi
            fi
        else
            echo "  Skipping WebP version (cached as no benefit)"
        fi

        # Create resized WebP versions from the original non-WebP source
        for width_webp in "${IMAGE_WIDTHS[@]}"; do
            # Only process if original is larger than target width
            if [ "$original_width" -gt "$width_webp" ]; then
                local webp_resized_target_file="$target_dir/${basename}-${width_webp}.webp"
                local variant_key="webp-${width_webp}"

                # Skip if this variant was previously processed but resulted in larger file
                if is_variant_skipped "$rel_path" "$variant_key"; then
                    echo "  Skipping ${width_webp}px WebP (cached as larger than original)"
                    continue
                fi

                # Check if this webp_resized_target_file needs processing based on the original source
                if needs_processing "$source" "$webp_resized_target_file"; then
                    echo "  Creating ${width_webp}px WebP version"
                    did_any_work=true
                    # Convert $source to webp_resized_target_file with resize and 90% quality
                    magick "$source" -resize "${width_webp}x>" -quality 90 "$webp_resized_target_file"

                    # Check if the processed WebP image is larger than the original source file
                    local processed_webp_size=$(get_file_size "$webp_resized_target_file")
                    echo "    ${width_webp}px WebP version size: $processed_webp_size bytes"
                    if [ "$processed_webp_size" -gt "$original_size" ]; then
                        echo "  Warning: ${width_webp}px WebP version is larger than original, removing and caching"
                        rm "$webp_resized_target_file"
                        add_skipped_variant "$rel_path" "$variant_key"
                    fi
                fi
            fi
        done
    fi
}

# Main function to process all images
process_all_images() {
    # Load cache from file
    echo "Loading image processing cache..."
    load_cache
    local cached_count=${#CACHE_MTIME[@]}
    echo "  Loaded $cached_count cached entries"

    # Find all images in the source directory
    echo "Finding images in $SOURCE_DIR..."
    local image_count=0
    local processed_count=0
    local skipped_count=0

    # Use a temp file to collect image paths since pipe creates subshell
    local temp_file=$(mktemp)
    find "$SOURCE_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) -not -path "*/processed/*" > "$temp_file"

    local total_images=$(wc -l < "$temp_file" | tr -d ' ')
    echo "Found $total_images images to check"

    while IFS= read -r image; do
        local rel_path="${image#$SOURCE_DIR/}"
        local current_mtime=$(get_mtime "$image")

        # Quick check: if in cache with same mtime, likely can skip
        if ! needs_processing_cached "$rel_path" "$current_mtime"; then
            skipped_count=$((skipped_count + 1))
            # Only show message every 100 skipped images to reduce noise
            if [ $((skipped_count % 100)) -eq 0 ]; then
                echo "  Skipped $skipped_count cached images so far..."
            fi
        else
            process_image "$image"
            processed_count=$((processed_count + 1))
        fi
        image_count=$((image_count + 1))
    done < "$temp_file"

    rm -f "$temp_file"

    # Save updated cache
    echo "Saving image processing cache..."
    save_cache

    echo ""
    echo "=== Image preprocessing summary ==="
    echo "  Total images found: $image_count"
    echo "  Skipped (cached): $skipped_count"
    echo "  Processed: $processed_count"
    echo "  Cache entries: ${#CACHE_MTIME[@]}"
    echo "Image preprocessing complete!"
}

# If script is run directly, process all images
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    process_all_images
fi
