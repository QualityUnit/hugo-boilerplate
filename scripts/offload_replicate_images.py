import os
import re
import requests
import toml
from pathlib import Path
from urllib.parse import urlparse
import random

CONTENT_DIR = Path(__file__).parents[3] / 'content'
STATIC_IMAGES_DIR = Path(__file__).parents[3] / 'static' / 'images'

IMG_URL_PREFIXES = [
    'http://',
    'https://',
]
IMG_PATTERN = re.compile(
    r'!\[([^\]]*)\]\(((?:' + '|'.join(re.escape(p) for p in IMG_URL_PREFIXES) + r')[^\s)]+)(?:\s+"([^"]+)")?\)'
)
TITLE_PATTERN = re.compile(r'title:\s*"([^"]+)"', re.IGNORECASE)

# Configurable list of image attributes to offload
# - string: simple attribute (e.g. 'image', 'originalCharacterImage')
# - dict: array attribute, key is array name, value is image key (e.g. {'characterImages': 'image'})
IMAGE_ATTRIBUTES = [
    "image",
    "originalCharacterImage",
    {"characterImages": "image"},
    # Add more as needed
]

# Configurable list of shortcodes and their image attributes to process
# Each entry specifies:
# - name: the shortcode name
# - attributes: list of attribute names that contain image URLs
# - use_alt: name of attribute to use for alt text (default: 'imageAlt')
SHORTCODE_IMAGE_ATTRIBUTES = [
    {
        "name": "lazyimg",
        "attributes": ["src"],
        "use_alt": "alt"
    },
    {
        "name": "features-with-fading-image",
        "attributes": ["imageUrl"],
        "use_alt": "imageAlt"
    },
    # Add more shortcodes as needed, for example:
    # {
    #     "name": "hero-banner",
    #     "attributes": ["backgroundImg", "logoImg"],
    #     "use_alt": "altText"
    # },
]

def url_matches_prefix(url):
    return any(url.startswith(prefix) for prefix in IMG_URL_PREFIXES)

def get_effective_rel_folder(md_path):
    rel_folder = md_path.parent.relative_to(CONTENT_DIR)
    parts = list(rel_folder.parts)
    # Remove language directory if present (e.g., 'en')
    if parts and len(parts[0]) == 2:  # crude check for language code
        parts = parts[1:]
    return Path(*parts)

def find_title_near_line(lines, idx):
    # Search upwards for a title attribute
    for i in range(idx, -1, -1):
        match = TITLE_PATTERN.search(lines[i])
        if match:
            return match.group(1)
    return 'untitled'

def process_image_url(url, out_dir, title, md_stem, idx=None):
    # 1. Determine extension
    ext = os.path.splitext(urlparse(url).path)[1]
    if not ext:
        try:
            resp = requests.head(url, allow_redirects=True)
            content_type = resp.headers.get('Content-Type', '')
            if 'image/' in content_type:
                ext = '.' + content_type.split('image/')[1].split(';')[0].strip().split('+')[0]
            else:
                ext = '.jpg'  # fallback
        except Exception as e:
            print(f"!!! ERROR getting extension for image {url}: {e}")
            ext = '.jpg'

    # 2. Try to use filename from URL if it's unique
    out_filename = None
    url_filename_str = os.path.basename(urlparse(url).path)
    if url_filename_str:
        # Sanitize filename from URL
        base, _ = os.path.splitext(url_filename_str)
        safe_base = re.sub(r'[^a-zA-Z0-9_-]', '_', base)
        candidate_filename = f"{safe_base}{ext}"
        if not (out_dir / candidate_filename).exists():
            out_filename = candidate_filename

    # 3. If filename from URL is not used, create a fallback name using page context
    if not out_filename:
        safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', title)
        if idx is not None:
            out_filename = f"{md_stem}_{safe_title}_{idx}{ext}"
        else:
            out_filename = f"{md_stem}_{safe_title}{ext}"
    #if filename contains more dash symbols, replace them with one ... e.g. --- > -
    out_filename = re.sub(r'-+', '-', out_filename)[-100:]

    # 4. Download image if it doesn't exist
    out_path = out_dir / out_filename
    if not out_path.exists():
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            with open(out_path, 'wb') as imgf:
                imgf.write(resp.content)
        except Exception as e:
            print(f"!!! ERROR downloading image from {url}: {e}")
            return None
    return out_filename

def process_md_file(md_path):
    rel_folder = get_effective_rel_folder(md_path)
    md_stem = md_path.stem  # Get the name of the md file without extension
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Split TOML frontmatter and body
    if content.startswith('+++'):
        end_idx = content.find('+++', 3)
        if end_idx != -1:
            toml_section = content[3:end_idx].strip()
            body = content[end_idx+3:]
        else:
            toml_section = ''
            body = content
    else:
        toml_section = ''
        body = content
    changed = False
    toml_changed = False
    body_changed = False
    # Parse TOML frontmatter
    if toml_section:
        try:
            data = toml.loads(toml_section)
        except toml.TomlDecodeError as e:
            print(f"!!! ERROR decoding TOML in {md_path}: {e}")
            return
        # Generic image attribute offloading
        for attr in IMAGE_ATTRIBUTES:
            if isinstance(attr, str):
                # Simple attribute
                if attr in data and isinstance(data[attr], str) and url_matches_prefix(data[attr]):
                    url = data[attr]
                    orig_title = data.get('title') or attr
                    base_title = orig_title if orig_title and orig_title.strip() else attr
                    out_dir = STATIC_IMAGES_DIR / rel_folder
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_filename_result = process_image_url(url, out_dir, base_title, md_stem)
                    if out_filename_result:
                        local_url = f"/images/{rel_folder}/{out_filename_result}".replace('\\', '/')
                        data[attr] = local_url
                        toml_changed = True
                        changed = True
            elif isinstance(attr, dict):
                # Array or dict attribute
                for arr_name, img_key in attr.items():
                    if arr_name in data and isinstance(data[arr_name], list):
                        for idx_entry, entry in enumerate(data[arr_name]):
                            if img_key in entry and isinstance(entry[img_key], str) and url_matches_prefix(entry[img_key]):
                                url = entry[img_key]
                                entry_title = entry.get('title') or img_key
                                base_title = entry_title if entry_title and entry_title.strip() else img_key
                                out_dir = STATIC_IMAGES_DIR / rel_folder
                                out_dir.mkdir(parents=True, exist_ok=True)
                                out_filename_result = process_image_url(url, out_dir, base_title, md_stem, idx=idx_entry)
                                if out_filename_result:
                                    local_url = f"/images/{rel_folder}/{out_filename_result}".replace('\\', '/')
                                    entry[img_key] = local_url
                                    toml_changed = True
                                    changed = True
                    elif arr_name in data and isinstance(data[arr_name], dict):
                        entry = data[arr_name]
                        if img_key in entry and isinstance(entry[img_key], str) and url_matches_prefix(entry[img_key]):
                            url = entry[img_key]
                            entry_title = entry.get('title') or img_key
                            base_title = entry_title if entry_title and entry_title.strip() else img_key
                            out_dir = STATIC_IMAGES_DIR / rel_folder
                            out_dir.mkdir(parents=True, exist_ok=True)
                            out_filename_result = process_image_url(url, out_dir, base_title, md_stem)
                            if out_filename_result:
                                local_url = f"/images/{rel_folder}/{out_filename_result}".replace('\\', '/')
                                entry[img_key] = local_url
                                toml_changed = True
                                changed = True
        if toml_changed:
            new_toml = toml.dumps(data)
        else:
            new_toml = toml_section
    else:
        new_toml = toml_section

    # Process body images
    lines = body.splitlines()
    for idx, line in enumerate(lines):
        # Markdown image syntax
        modified_line = line
        for match_idx, match in enumerate(IMG_PATTERN.finditer(line)):
            alt_text = match.group(1)  # Get alt text inside []
            url = match.group(2)
            explicit_title = match.group(3)  # Title after URL in quotes

            if not url_matches_prefix(url):
                continue

            # Priority: 1. Explicit title in quotes, 2. Alt text, 3. Title from nearby TOML
            title = explicit_title or alt_text or find_title_near_line(lines, idx)
            if not title or title.strip() == '':
                title = 'untitled'

            out_dir = STATIC_IMAGES_DIR / rel_folder
            out_dir.mkdir(parents=True, exist_ok=True)

            out_filename_result = process_image_url(url, out_dir, title, md_stem, idx=f"md{idx}_{match_idx}")

            if out_filename_result:
                local_url = f"/images/{rel_folder}/{out_filename_result}".replace('\\', '/')
                new_img_md = f'![{alt_text}]({local_url} "{title}")'
                modified_line = modified_line.replace(match.group(0), new_img_md)
                body_changed = True
                changed = True
        lines[idx] = modified_line

    # Combine all lines back to body content
    body = '\n'.join(lines)

    # Process all shortcodes using the general method
    body, sc_changed = process_shortcodes(body, rel_folder, md_stem)
    if sc_changed:
        body_changed = True
        changed = True

    if changed:
        # Write both TOML and body together, always
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"+++\n{new_toml}\n+++\n{body}")

def process_shortcodes(content, rel_folder, md_stem):
    """
    Process all shortcodes defined in SHORTCODE_IMAGE_ATTRIBUTES
    Returns: (modified_content, changed_flag)
    """
    modified_content = content
    content_changed = False

    for shortcode_config in SHORTCODE_IMAGE_ATTRIBUTES:
        shortcode_name = shortcode_config["name"]
        image_attributes = shortcode_config["attributes"]
        alt_attribute = shortcode_config.get("use_alt", "imageAlt")

        # Create a pattern that matches this shortcode and captures its contents
        # Works for both single-line and multi-line shortcodes
        pattern = re.compile(
            r'\{\{<\s*' + re.escape(shortcode_name) + r'\s+([^>]*?)>}}',
            re.DOTALL
        )

        # Find all instances of this shortcode
        replacements = []
        for match_idx, match in enumerate(pattern.finditer(modified_content)):
            shortcode_content = match.group(0)
            attributes_text = match.group(1)

            # Process each image attribute in this shortcode
            new_shortcode = shortcode_content
            for attr_name in image_attributes:
                # Look for the attribute in the shortcode content
                attr_pattern = re.compile(r'(?:^|\s)' + re.escape(attr_name) + r'="([^"]+)"')
                attr_match = attr_pattern.search(attributes_text)

                if attr_match:
                    url = attr_match.group(1)
                    if url_matches_prefix(url):
                        # Get alt text if it exists
                        alt_pattern = re.compile(r'(?:^|\s)' + re.escape(alt_attribute) + r'="([^"]+)"')
                        alt_match = alt_pattern.search(attributes_text)
                        alt_text = alt_match.group(1) if alt_match else f"{shortcode_name}_{match_idx}"

                        # Process the image
                        out_dir = STATIC_IMAGES_DIR / rel_folder
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_filename_result = process_image_url(
                            url, out_dir, alt_text, md_stem,
                            idx=f"{shortcode_name}{match_idx}_{attr_name}"
                        )

                        if out_filename_result:
                            local_url = f"/images/{rel_folder}/{out_filename_result}".replace('\\', '/')
                            # Replace the URL in the attribute
                            new_attr = f'{attr_name}="{local_url}"'
                            old_attr = f'{attr_name}="{url}"'
                            new_shortcode = new_shortcode.replace(old_attr, new_attr)
                            content_changed = True

            if new_shortcode != shortcode_content:
                replacements.append((shortcode_content, new_shortcode))

        # Apply all replacements
        for old, new in replacements:
            modified_content = modified_content.replace(old, new)

    return modified_content, content_changed

def main():
    for root, _, files in os.walk(CONTENT_DIR):
        for file in files:
            if file.endswith('.md'):
                process_md_file(Path(root) / file)

if __name__ == '__main__':
    main()