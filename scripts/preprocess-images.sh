#!/bin/bash

# Image preprocessing script for Wachman
# This script preprocesses images and stores them in the static/images/processed directory
# Images are only processed if they are larger than the size limits
# For WebP images, only size alternatives are created, no format conversion
# If the processed image is larger than the original, it is not used

# Set error handling
set -e

# Get the root directory of the Hugo site
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
THEME_DIR="$(dirname "$SCRIPT_DIR")"
HUGO_ROOT="$(dirname "$(dirname "$THEME_DIR")")"

SOURCE_DIR="$HUGO_ROOT/static/images"
TARGET_DIR="$HUGO_ROOT/static/images/processed"

# Create the processed directory if it doesn't exist
mkdir -p "$TARGET_DIR"

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
    
    echo "Processing: $rel_path"
    
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
    if needs_processing "$source" "$optimized_original"; then
        echo "  Creating optimized original in $target_dir"
        
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
    
    # Process each width for the image
    for width in "${IMAGE_WIDTHS[@]}"; do
        # Only process if original is larger than target width
        if [ "$original_width" -gt "$width" ]; then
            local target_file="$target_dir/${basename}-${width}.${extension}"
            
            if needs_processing "$source" "$target_file"; then
                echo "  Creating ${width}px width"
                
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
                    echo "  Warning: ${width}px version is larger than original, removing"
                    rm "$target_file"
                fi
            fi
        fi
    done
    
    # For non-WebP images, create a WebP version of the original
    if [ "$is_webp" = false ]; then
        local webp_target="$target_dir/${basename}.webp"
        
        if needs_processing "$source" "$webp_target"; then
            echo "  Creating WebP version"
            magick "$source" -quality "$QUALITY_WEBP" "$webp_target"
            
            # Log the WebP file size but keep it even if larger
            local processed_size=$(get_file_size "$webp_target")
            echo "    WebP version size: $processed_size bytes"
            if [ "$processed_size" -gt "$original_size" ]; then
                echo "  Note: WebP version is larger than original, but keeping it for browser compatibility"
            fi
        fi

        # Create resized WebP versions from the original non-WebP source
        for width_webp in "${IMAGE_WIDTHS[@]}"; do
            # Only process if original is larger than target width
            if [ "$original_width" -gt "$width_webp" ]; then
                local webp_resized_target_file="$target_dir/${basename}-${width_webp}.webp"

                # Check if this webp_resized_target_file needs processing based on the original source
                if needs_processing "$source" "$webp_resized_target_file"; then
                    echo "  Creating ${width_webp}px WebP version"
                    # Convert $source to webp_resized_target_file with resize and 90% quality
                    magick "$source" -resize "${width_webp}x>" -quality 90 "$webp_resized_target_file"

                    # Check if the processed WebP image is larger than the original source file
                    local processed_webp_size=$(get_file_size "$webp_resized_target_file")
                    echo "    ${width_webp}px WebP version size: $processed_webp_size bytes"
                    if [ "$processed_webp_size" -gt "$original_size" ]; then
                        echo "  Warning: ${width_webp}px WebP version is larger than original, removing"
                        rm "$webp_resized_target_file"
                    fi
                fi
            fi
        done
    fi
}

# Main function to process all images
process_all_images() {
    # Find all images in the source directory
    echo "Finding images in $SOURCE_DIR..."
    image_count=0

    find "$SOURCE_DIR" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) -not -path "*/processed/*" | while read -r image; do
        process_image "$image"
        image_count=$((image_count + 1))
    done

    echo "Processed $image_count images"
    echo "Image preprocessing complete!"
}

# If script is run directly, process all images
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    process_all_images
fi
