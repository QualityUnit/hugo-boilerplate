#!/bin/bash

# Video preprocessing script
# Optimizes .mp4 files in cdn-assets/ using ffmpeg (libx264, CRF 23)
# Replaces originals only if the optimized version is smaller.
#
# Uses a cache file in data/ to skip already-processed videos.
#
# Usage: bash themes/boilerplate/scripts/preprocess-videos.sh

set -e

# Get the root directory of the Hugo site
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
THEME_DIR="$(dirname "$SCRIPT_DIR")"
HUGO_ROOT="$(dirname "$(dirname "$THEME_DIR")")"

SOURCE_DIR="$HUGO_ROOT/cdn-assets"
CACHE_FILE="$HUGO_ROOT/data/video_optimization_cache.json"

# Check ffmpeg is available
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg not found. Install with: brew install ffmpeg"
    exit 1
fi

# Initialize cache file if it doesn't exist
if [ ! -f "$CACHE_FILE" ]; then
    echo '{}' > "$CACHE_FILE"
fi

# Get file modification time in seconds since epoch
get_mtime() {
    stat -f%m "$1" 2>/dev/null || stat -c%Y "$1" 2>/dev/null || echo "0"
}

# Get file size in bytes
get_file_size() {
    stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo "0"
}

# Check if video is in cache with same mtime
is_cached() {
    local rel_path="$1"
    local current_mtime="$2"
    local cached_mtime

    cached_mtime=$(python3 -c "
import json, sys
try:
    with open('$CACHE_FILE') as f:
        cache = json.load(f)
    entry = cache.get('$rel_path', {})
    print(entry.get('mtime', ''))
except:
    print('')
" < /dev/null 2>/dev/null)

    [ "$cached_mtime" = "$current_mtime" ]
}

# Update cache entry
update_cache() {
    local rel_path="$1"
    local mtime="$2"
    local size="$3"

    python3 -c "
import json
try:
    with open('$CACHE_FILE') as f:
        cache = json.load(f)
except:
    cache = {}
cache['$rel_path'] = {'mtime': $mtime, 'size': $size}
with open('$CACHE_FILE', 'w') as f:
    json.dump(cache, f, indent=2)
" < /dev/null 2>/dev/null
}

# Main
echo "Finding videos in $SOURCE_DIR..."

# Collect all video paths into a temp file first to avoid stdin conflicts
# (ffmpeg/python3 consume stdin from pipe, breaking while-read loops)
TEMP_FILE=$(mktemp)
find "$SOURCE_DIR" -type f -name "*.mp4" -not -path "*/processed/*" | sort > "$TEMP_FILE"

optimized=0
skipped=0
total=0

while IFS= read -r video; do
    [ -z "$video" ] && continue
    rel_path="${video#$SOURCE_DIR/}"
    current_mtime=$(get_mtime "$video")
    current_size=$(get_file_size "$video")
    total=$((total + 1))

    # Skip if cached
    if is_cached "$rel_path" "$current_mtime"; then
        skipped=$((skipped + 1))
        continue
    fi

    echo "  Processing: $rel_path ($(echo "scale=1; $current_size / 1024 / 1024" | bc)MB)"

    # Optimize MP4 in-place
    tmp_path="${video}.tmp.mp4"

    if ffmpeg -i "$video" -c:v libx264 -crf 23 -preset slow -an -y "$tmp_path" </dev/null 2>/dev/null; then
        new_size=$(get_file_size "$tmp_path")

        if [ "$new_size" -lt "$current_size" ]; then
            mv "$tmp_path" "$video"
            saved=$(echo "scale=0; (1 - $new_size / $current_size) * 100" | bc)
            echo "    ✓ MP4: $((current_size / 1024))KB → $((new_size / 1024))KB (${saved}% smaller)"
        else
            rm -f "$tmp_path"
            echo "    ⏭ MP4 already optimal"
        fi
    else
        rm -f "$tmp_path"
        echo "    ❌ MP4 optimization failed"
    fi

    # Update cache
    final_mtime=$(get_mtime "$video")
    final_size=$(get_file_size "$video")
    update_cache "$rel_path" "$final_mtime" "$final_size"
    optimized=$((optimized + 1))

done < "$TEMP_FILE"

rm -f "$TEMP_FILE"

echo ""
echo "=== Video preprocessing summary ==="
echo "  Total videos found: $total"
echo "  Skipped (cached): $skipped"
echo "  Processed: $optimized"
echo "Video preprocessing complete!"
