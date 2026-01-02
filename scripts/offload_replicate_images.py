import os
import re
import requests
import tomllib
from pathlib import Path
from toml_frontmatter import safe_toml_dumps
from urllib.parse import urlparse, unquote
import random
from dotenv import load_dotenv

# Load environment variables from .env file in the same directory as this script
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

# FlowHunt S3 configuration
FLOWHUNT_S3_BUCKET = os.getenv('FLOWHUNT_S3_BUCKET')
FLOWHUNT_AWS_ACCESS_KEY_ID = os.getenv('FLOWHUNT_AWS_ACCESS_KEY_ID')
FLOWHUNT_AWS_SECRET_ACCESS_KEY = os.getenv('FLOWHUNT_AWS_SECRET_ACCESS_KEY')
FLOWHUNT_AWS_REGION = os.getenv('FLOWHUNT_AWS_REGION', 'eu-central-1')

# Global flag to track user's choice for FlowHunt images
_flowhunt_user_choice = None  # None = not asked, 'skip' = skip all, 'configure' = user wants to configure

# FlowHunt S3 URL pattern
FLOWHUNT_S3_PATTERN = re.compile(r'https?://flowhunt-photo-ai\.s3\.amazonaws\.com/(.+?)(?:\?|$)')


def is_flowhunt_s3_url(url):
    """Check if URL is from FloatHunt S3 bucket."""
    return 'flowhunt-photo-ai.s3.amazonaws.com' in url


def extract_s3_path(url):
    """Extract the S3 object path from a FlowHunt S3 URL."""
    match = FLOWHUNT_S3_PATTERN.match(url)
    if match:
        return unquote(match.group(1))
    return None


def flowhunt_credentials_configured():
    """Check if FlowHunt S3 credentials are properly configured."""
    return all([
        FLOWHUNT_S3_BUCKET,
        FLOWHUNT_AWS_ACCESS_KEY_ID,
        FLOWHUNT_AWS_SECRET_ACCESS_KEY
    ])


def ask_user_flowhunt_config():
    """Ask user what to do when FlowHunt credentials are not configured."""
    global _flowhunt_user_choice

    if _flowhunt_user_choice is not None:
        return _flowhunt_user_choice

    print("\n" + "=" * 60)
    print("FlowHunt S3 credentials not configured!")
    print("=" * 60)
    print("\nImages from flowhunt-photo-ai.s3.amazonaws.com cannot be")
    print("downloaded because they have expired signed URLs.")
    print("\nTo fix this, add the following to your .env file:")
    print("  FLOWHUNT_S3_BUCKET=flowhunt-photo-ai")
    print("  FLOWHUNT_AWS_ACCESS_KEY_ID=your-access-key")
    print("  FLOWHUNT_AWS_SECRET_ACCESS_KEY=your-secret-key")
    print("  FLOWHUNT_AWS_REGION=eu-central-1")
    print("\nWhat would you like to do?")
    print("  [s] Skip all FlowHunt images for now")
    print("  [q] Quit so I can configure credentials")

    while True:
        choice = input("\nYour choice (s/q): ").strip().lower()
        if choice == 's':
            _flowhunt_user_choice = 'skip'
            print("Skipping FlowHunt images...")
            return 'skip'
        elif choice == 'q':
            _flowhunt_user_choice = 'configure'
            return 'configure'
        else:
            print("Please enter 's' to skip or 'q' to quit.")


def download_from_flowhunt_s3(s3_path, out_path):
    """Download a file directly from FlowHunt S3 bucket using boto3."""
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        print("!!! ERROR: boto3 not installed. Run: pip install boto3")
        return False

    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=FLOWHUNT_AWS_ACCESS_KEY_ID,
            aws_secret_access_key=FLOWHUNT_AWS_SECRET_ACCESS_KEY,
            region_name=FLOWHUNT_AWS_REGION
        )

        s3_client.download_file(FLOWHUNT_S3_BUCKET, s3_path, str(out_path))
        print(f"    Downloaded from S3: {s3_path}")
        return True
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        print(f"!!! ERROR downloading from S3 ({error_code}): {s3_path}")
        return False
    except Exception as e:
        print(f"!!! ERROR downloading from S3: {e}")
        return False


CONTENT_DIR = Path(__file__).parents[3] / 'content'
STATIC_IMAGES_DIR = Path(__file__).parents[3] / 'cdn-assets'

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
    "image", "screenshot",
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

def get_language_directories():
    """Get list of language directories in content folder."""
    if not CONTENT_DIR.exists():
        return set()
    return {d.name for d in CONTENT_DIR.iterdir() if d.is_dir()}

# Cache language directories at module load
LANGUAGE_DIRS = get_language_directories()

def get_effective_rel_folder(md_path):
    rel_folder = md_path.parent.relative_to(CONTENT_DIR)
    parts = list(rel_folder.parts)
    # Remove language directory if present (e.g., 'en', 'pt-br', etc.)
    if parts and parts[0] in LANGUAGE_DIRS:
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
            resp = requests.head(url, allow_redirects=True, timeout=5)
            content_type = resp.headers.get('Content-Type', '')
            if 'image/' in content_type:
                ext = '.' + content_type.split('image/')[1].split(';')[0].strip().split('+')[0]
            else:
                ext = '.jpg'  # fallback
        except Exception as e:
            print(f"!!! ERROR getting extension for image {url}: {e}")
            ext = '.jpg'

    # 2. Generate filename from URL
    out_filename = None
    url_filename_str = os.path.basename(urlparse(url).path)
    if url_filename_str:
        # Sanitize filename from URL
        base, _ = os.path.splitext(url_filename_str)
        safe_base = re.sub(r'[^a-zA-Z0-9_-]', '_', base)
        out_filename = f"{safe_base}{ext}"

    # 3. If no filename from URL, create a fallback name using page context
    if not out_filename:
        safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', title)
        if idx is not None:
            out_filename = f"{md_stem}_{safe_title}_{idx}{ext}"
        else:
            out_filename = f"{md_stem}_{safe_title}{ext}"

    # Normalize filename: replace multiple dashes with one, limit length
    out_filename = re.sub(r'-+', '-', out_filename)[-100:]

    # 4. Check if file already exists - if so, just return the path (no download needed)
    out_path = out_dir / out_filename
    if out_path.exists():
        return out_filename

    # 5. Download image since it doesn't exist locally
    download_success = False

    # First, try regular HTTP download
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        # Check if the response is actually an image
        content_type = resp.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            print(f"!!! ERROR: URL {url} returned HTML instead of an image")
        elif resp.content.startswith(b'<!DOCTYPE') or resp.content.startswith(b'<html'):
            print(f"!!! ERROR: URL {url} returned HTML content instead of an image")
        else:
            with open(out_path, 'wb') as imgf:
                imgf.write(resp.content)
            download_success = True
    except requests.exceptions.HTTPError as e:
        # Check if it's a 403 error from FlowHunt S3
        if e.response.status_code == 403 and is_flowhunt_s3_url(url):
            print(f"    FlowHunt S3 URL returned 403, attempting S3 direct download...")

            if flowhunt_credentials_configured():
                s3_path = extract_s3_path(url)
                if s3_path:
                    download_success = download_from_flowhunt_s3(s3_path, out_path)
            else:
                # Ask user what to do
                user_choice = ask_user_flowhunt_config()
                if user_choice == 'configure':
                    print("\nExiting. Please configure FlowHunt credentials and run again.")
                    import sys
                    sys.exit(1)
                # 'skip' - just continue without downloading
        else:
            print(f"!!! ERROR downloading image from {url}: {e}")
    except Exception as e:
        print(f"!!! ERROR downloading image from {url}: {e}")

    if not download_success:
        return None
    return out_filename

def process_md_file(md_path):
    try:
        rel_folder = get_effective_rel_folder(md_path)
        md_stem = md_path.stem  # Get the name of the md file without extension
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"!!! ERROR: File not found: {md_path}")
        return
    except Exception as e:
        print(f"!!! ERROR: Failed to read file {md_path}: {e}")
        return
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
            data = tomllib.loads(toml_section)
        except tomllib.TOMLDecodeError as e:
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
            new_toml = safe_toml_dumps(data)
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
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"+++\n{new_toml}\n+++\n{body}")
        except Exception as e:
            print(f"!!! ERROR: Failed to write file {md_path}: {e}")

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