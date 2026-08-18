#!/usr/bin/env python3
"""
translate_with_flowhunt.py

This script translates files from /content/en/* to all language variations defined in /content/[lang]/
that don't already exist in the target language directories using FlowHunt API with Flow Sessions.

The script uses the new FlowHunt SDK with the following workflow:
1. Create a flow session with variables (language, filename)
2. Upload the file as an attachment to the session
3. Invoke the translation task in the session
4. Monitor session events until a file artifact with translation URL is received
5. Download the translated file and save it with the correct filename

Usage:
    python translate_with_flowhunt.py [--path /path/to/content] [--check-interval 5] [--flow-id FLOW_ID] [--max-scheduled-tasks LIMIT]

Prerequisites:
    - Python 3.6 or higher
    - FlowHunt API key (set in .env file or as environment variable FLOWHUNT_API_KEY)
    - Required packages: flowhunt, tqdm, python-dotenv, requests

Examples:
    # Basic usage (will use ../content/ relative to the script location)
    python translate_with_flowhunt.py

    # With explicit path
    python translate_with_flowhunt.py --path /Users/username/work/hugo-boilerplate/content

    # With custom flow and workspace IDs
    python translate_with_flowhunt.py --flow-id "custom-flow-id"

    # With maximum batch size of 100 scheduled tasks
    python translate_with_flowhunt.py --max-scheduled-tasks 100

    # With API key as environment variable
    export FLOWHUNT_API_KEY="your-api-key"
    python translate_with_flowhunt.py
"""

import os
import re
import sys
import argparse
import time
import json
import requests
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import flowhunt
from pprint import pprint
from functools import wraps

# TOML reader for validating translated frontmatter. tomllib is stdlib from
# 3.11 (the version the translate-content workflow pins); the `toml` package is
# already in requirements.txt and covers anyone running this on an older local
# interpreter.
try:
    import tomllib

    def _toml_loads(text):
        return tomllib.loads(text)
except ImportError:  # Python < 3.11
    import toml

    def _toml_loads(text):
        return toml.loads(text)

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
hugo_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))  # Adjusted to point to the correct root
env_path = os.path.join(script_dir, '.env')
if os.path.exists(env_path):
    print(f"Loading environment variables from {env_path}")
    load_dotenv(env_path)
else:
    print("No .env file found, using environment variables if available")

# Get API key from environment variable
api_key = os.getenv("FLOWHUNT_API_KEY")
if not api_key:
    print("Error: FLOWHUNT_API_KEY not found in environment variables or .env file")
    print("Please set the FLOWHUNT_API_KEY environment variable or add it to the .env file")
    sys.exit(1)

# Default FlowHunt flow ID for translation service (new session-based flow).
# Override with FLOWHUNT_FLOW_ID or --flow-id.
DEFAULT_FLOW_ID = os.getenv('FLOWHUNT_FLOW_ID', '9df82032-0c90-4a60-8538-5d724590562b')

# Workspace that owns the translation flow.
#
# FlowHunt API keys are scoped to a single workspace — a key issued in
# workspace A cannot act on workspace B, and every workspace-scoped call
# (create_flow_session, GET /v2/flows/{id}, …) answers 401 while unscoped ones
# like /v2/credits/balance still return 200. That asymmetry makes it look like
# a bad key when it is really a workspace mismatch.
#
# Hard-coding LiveAgentWP here forced every downstream project to use a key
# from that one workspace. Override with FLOWHUNT_WORKSPACE_ID or
# --workspace-id to point at the workspace your own key belongs to.
DEFAULT_WORKSPACE_ID = os.getenv('FLOWHUNT_WORKSPACE_ID', '70ff1135-5ce6-42a7-8abe-ec03f58e828e')

# Map of folder names to full language names
LANGUAGE_MAP = {
    # ISO 639-1 language codes
    'af': 'Afrikaans',
    'ar': 'Arabic',
    'bg': 'Bulgarian',
    'bn': 'Bengali',
    'ca': 'Catalan',
    'cs': 'Czech',
    'da': 'Danish',
    'de': 'German',
    'el': 'Greek',
    'en': 'English',
    'es': 'Spanish',
    'et': 'Estonian',
    'fa': 'Persian',
    'fi': 'Finnish',
    'fr': 'French',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hr': 'Croatian',
    'hu': 'Hungarian',
    'id': 'Indonesian',
    'is': 'Icelandic',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'ms': 'Malay',
    'nl': 'Dutch',
    'no': 'Norwegian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'ro': 'Romanian',
    'ru': 'Russian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'sq': 'Albanian',
    'sr': 'Serbian',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'ta': 'Tamil',
    'th': 'Thai',
    'tr': 'Turkish',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'vi': 'Vietnamese',
    'zh': 'Chinese',
    'us': 'American English',
    
    # Country-specific language codes
    'pt-br': 'Brazilian Portuguese',
    'zh-cn': 'Simplified Chinese',
    'zh-tw': 'Traditional Chinese',
    'en-gb': 'British English',
    'en-us': 'American English',
    'es-mx': 'Mexican Spanish',
    
    # Special cases that might be confused
    'ch': 'Swiss German',  # Not Chinese, but Swiss domain/German dialect
    'cy': 'Welsh',  # Not Cypriot
    'gl': 'Galician',  # Not Greenlandic
    'mt': 'Maltese',  # Not Montenegrin
    'eu': 'Basque',  # Not European Union
}



def retry_on_429(func, *args, max_retries=5, default_wait=2, **kwargs):
    """
    Call func(*args, **kwargs) and retry on 429 (Too Many Requests) errors.
    Uses the retry-after header when available, otherwise waits default_wait seconds.
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except flowhunt.exceptions.ApiException as e:
            if e.status == 429 and attempt < max_retries:
                # Parse retry-after from headers if available
                wait_time = default_wait
                if hasattr(e, 'headers') and e.headers:
                    retry_after = e.headers.get('retry-after')
                    if retry_after:
                        try:
                            wait_time = max(int(retry_after), 1)
                        except (ValueError, TypeError):
                            pass
                print(f"[RATE LIMIT] 429 received, waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                raise


def get_workspace_id(workspace_id=None):
    # If a workspace ID is provided, use it directly
    if workspace_id:
        return workspace_id

    # Use the default workspace ID for LiveAgent
    if DEFAULT_WORKSPACE_ID:
        return DEFAULT_WORKSPACE_ID

    # No fallback available - DEFAULT_WORKSPACE_ID must be set
    print("Error: No workspace ID available. DEFAULT_WORKSPACE_ID must be set.")
    return None
    

def is_translatable_file(file_path):
    """Check if a file should be translated based on extension"""
    return file_path.suffix.lower() in ['.md', '.markdown', '.yaml', '.yml', '.html', '.txt']


# ---------------------------------------------------------------------------
# Translated frontmatter validation
#
# The translation flow regularly returns TOML frontmatter Hugo cannot
# unmarshal. Two failure modes seen in production so far:
#
#   answer = "…viens pogas "Saglabāt" klikšķis…"   unescaped inner quote
#   question =="Kokio nukreipimo rodiklio…"        duplicated '='
#
# An unescaped quote terminates the basic string early, so Hugo aborts the
# whole language build ("unmarshal failed: toml: expected newline but got …").
# One bad page takes down all ~1500 pages of that language, and Hugo only
# reports the first error per language, so they surface one at a time.
#
# Repairing what is deterministically repairable keeps those pages translated;
# anything left over is written anyway and reported, because a broken file can
# be fixed by hand while a missing one has to be re-translated.
# ---------------------------------------------------------------------------

FRONTMATTER_DELIM = '+++'

# A single-line TOML assignment: key, one-or-more '=', gap, then the value.
_KEY_EQ_RE = re.compile(r'^(\s*[A-Za-z_][A-Za-z0-9_-]*\s*)(=+)(\s*)(.*)$')


def split_toml_frontmatter(text):
    """
    Split a Hugo file into (before, frontmatter, body) around the +++ fences.

    Returns None when the text does not open with a TOML frontmatter block —
    YAML frontmatter, .html and .txt files are simply not our business.
    """
    if not text.startswith(FRONTMATTER_DELIM):
        return None
    parts = text.split(FRONTMATTER_DELIM, 2)
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2]


def validate_toml_frontmatter(text):
    """
    Check the TOML frontmatter of a translated file.

    Returns None when the frontmatter parses (or when there is none to check),
    otherwise the parse error as a string.
    """
    split = split_toml_frontmatter(text)
    if split is None:
        return None
    try:
        _toml_loads(split[1])
        return None
    except Exception as e:
        return str(e)


def _repair_toml_line(line):
    """
    Try to repair one malformed `key = "value"` line.

    Handles a duplicated assignment operator and unescaped ASCII double quotes
    inside the value, including the case where the closing quote is missing
    altogether. Returns the repaired line, or None when no repair applies.
    """
    m = _KEY_EQ_RE.match(line)
    if not m:
        return None

    key, _eq, _gap, value = m.groups()

    # Multi-line basic strings are out of scope — never reflow them.
    if value.startswith('"""') or not value.startswith('"'):
        return None

    stripped = value.rstrip()
    trailing = value[len(stripped):]

    inner = stripped[1:]
    if inner.endswith('"'):
        inner = inner[:-1]

    # Escape every double quote that is not escaped already. A missing closing
    # quote is handled implicitly: inner simply runs to end of line and gets a
    # fresh closing quote below.
    inner = re.sub(r'(?<!\\)"', r'\\"', inner)

    return f'{key.rstrip()} = "{inner}"{trailing}'


def repair_toml_frontmatter(text):
    """
    Attempt to repair malformed TOML frontmatter.

    Only lines that fail to parse on their own are touched, and a repair is
    kept only if the repaired line parses. The rebuilt frontmatter is then
    parsed as a whole, so a repair that fixes one line while breaking the
    document is discarded rather than written.

    Returns (repaired_text, repaired_line_numbers), or (None, []) when the
    frontmatter could not be made to parse.
    """
    split = split_toml_frontmatter(text)
    if split is None:
        return None, []
    before, frontmatter, body = split

    lines = frontmatter.split('\n')
    repaired_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Blank lines, comments and table headers are never the problem, and
        # array/multi-line continuations must not be parsed in isolation.
        if not stripped or stripped.startswith('#') or stripped.startswith('['):
            continue
        try:
            _toml_loads(stripped)
            continue  # this line is fine by itself
        except Exception:
            pass

        fixed = _repair_toml_line(line)
        if fixed is None or fixed == line:
            continue
        try:
            _toml_loads(fixed.strip())
        except Exception:
            continue  # repair did not help — leave the original alone

        lines[i] = fixed
        repaired_lines.append(i + 1)

    if not repaired_lines:
        return None, []

    candidate = '\n'.join(lines)
    try:
        _toml_loads(candidate)
    except Exception:
        return None, []

    return before + FRONTMATTER_DELIM + candidate + FRONTMATTER_DELIM + body, repaired_lines


def get_target_languages(content_dir):
    """
    Find all language directories in the content directory except 'en'
    
    Args:
        content_dir (Path): Path to the content directory
        
    Returns:
        list: List of target language directory names
    """
    target_langs = []
    
    for item in content_dir.iterdir():
        if item.is_dir() and item.name != "en":
            target_langs.append(item.name)
            
    return target_langs

def initialize_api_client():
    """Initialize and return a FlowHunt API client"""
    configuration = flowhunt.Configuration(
        host="https://api.flowhunt.io"
    )
    configuration.api_key['APIKeyHeader'] = api_key
    
    return flowhunt.ApiClient(configuration)

def create_translation_session(api_instance, file_path, content, target_lang, flow_id, workspace_id):
    """
    Create a FlowHunt flow session for translation using the new session-based API

    This function follows the new workflow:
    1. Create a flow session with variables (language, filename)
    2. Upload the file as an attachment to the session
    3. Invoke the translation task in the session

    Args:
        api_instance: FlowHunt API instance
        file_path (Path): Path to the source file
        content (str): Content to translate
        target_lang (str): Target language code
        flow_id (str): FlowHunt flow ID
        workspace_id (str): FlowHunt workspace ID

    Returns:
        dict: Session info containing session_id, from_timestamp, and start_time, or None if failed
    """
    try:
        # Get the full language name from the map, fallback to the code if not found
        language_name = LANGUAGE_MAP.get(target_lang.lower(), target_lang)
        filename = file_path.name

        print(f"[DEBUG] Creating session for {filename} -> {target_lang}")

        # Step 1: Create flow session with variables
        from_flow_create_session_req = flowhunt.FlowSessionCreateFromFlowRequest(
            flow_id=flow_id,
            variables={
                "source_language": "English",
                "target_language": language_name,
                "filename": filename,
                "today": time.strftime("%Y-%m-%d %H:00:00"),
            }
        )

        create_session_rsp = retry_on_429(
            api_instance.create_flow_session,
            workspace_id=workspace_id,
            flow_session_create_from_flow_request=from_flow_create_session_req
        )

        session_id = create_session_rsp.session_id
        print(f"[DEBUG] Created session {session_id} for {filename}")

        # Step 2: Upload the file as an attachment to the session.
        # SDK 3.18.2's FlowsApi.upload_attachments mis-serializes the file as
        # a form text field (its `_files` dict is empty), but the server
        # expects a real multipart/form-data file upload (UploadFile).
        # Until the SDK is fixed upstream, post directly with `requests`.
        upload_url = f"https://api.flowhunt.io/v2/flows/sessions/{session_id}/attachments"
        upload_headers = {"Api-Key": api_key}
        upload_files = {"file": (filename, content.encode("utf-8"), "application/octet-stream")}
        for attempt in range(6):
            resp = requests.post(upload_url, headers=upload_headers, files=upload_files, timeout=60)
            if resp.status_code == 429 and attempt < 5:
                wait_time = max(int(resp.headers.get("retry-after", "2") or "2"), 1)
                print(f"[RATE LIMIT] 429 on attachment upload, waiting {wait_time}s (attempt {attempt + 1}/5)")
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            break
        print(f"[DEBUG] Uploaded file attachment for {filename} to session {session_id}")

        # Step 3: Invoke the translation task
        translation_message = f"Translate to {language_name}"

        invoke_rsp = retry_on_429(
            api_instance.invoke_flow_response,
            session_id=session_id,
            flow_session_invoke_request=flowhunt.FlowSessionInvokeRequest(
                message=translation_message
            )
        )

        print(f"[DEBUG] Invoked translation in session {session_id}")

        return {
            'session_id': session_id,
            'from_timestamp': str(invoke_rsp.created_at),
            'start_time': time.time()
        }

    except Exception as e:
        print(f"Error creating translation session for {target_lang}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def check_session_results(api_instance, session_info, timeout=600):
    """
    Check if a flow session has completed and get the translation.md file URL

    This function monitors session events until a file artifact named 'translation.md'
    with the URL to the translated content is received.

    Args:
        api_instance: FlowHunt API instance
        session_info (dict): Session info containing session_id, start_time, and from_timestamp
        timeout (int): Maximum time in seconds to wait for session completion (default: 600 = 10 minutes)

    Returns:
        tuple: (is_ready, file_url)
    """
    try:
        session_id = session_info['session_id']
        from_ts = session_info.get('from_timestamp', '0')

        # Check for timeout
        elapsed_time = time.time() - session_info['start_time']
        if elapsed_time > timeout:
            print(f"[ERROR] Session {session_id} timed out after {timeout} seconds")
            return True, None

        # Poll for flow response using raw response to avoid SDK validation issues
        resp = retry_on_429(
            api_instance.poll_flow_response_without_preload_content,
            session_id=session_id,
            from_timestamp=from_ts
        )
        raw = json.loads(resp.data.decode('utf-8'))

        # Normalize: the API may return a plain list or a paginated wrapper dict
        # (e.g. {"items": [...], "total": N}) — iterating a dict yields string keys
        # which breaks event.get() calls below.
        if isinstance(raw, list):
            events = raw
        elif isinstance(raw, dict):
            events = raw.get('items', raw.get('data', []))
        else:
            events = []

        # Update timestamp for next poll
        for event in events:
            if not isinstance(event, dict):
                continue

            ts = event.get('created_at_timestamp')
            if ts:
                try:
                    session_info['from_timestamp'] = str(int(ts) + 1)
                except (ValueError, TypeError):
                    pass

            action_type = event.get('action_type')
            metadata = event.get('metadata') or {}
            if not isinstance(metadata, dict):
                metadata = {}

            # Check for artefacts with translation file
            if action_type == 'artefacts':
                artefacts = metadata.get('artefacts', [])
                for art in artefacts:
                    if not isinstance(art, dict):
                        continue
                    name = art.get('name', '')
                    url = art.get('download_url', '')
                    if 'translation' in name.lower() and url:
                        return True, url

            # Check for failure
            if action_type == 'failed':
                print(f"[ERROR] Session {session_id} failed")
                return True, None

        # Not ready yet
        return False, None

    except Exception as e:
        # Don't print errors on every check to avoid spam
        if 'start_time' in session_info and time.time() - session_info['start_time'] > 60:
            print(f"Error checking session results for {session_info.get('session_id')}: {str(e)}")
        return False, None


def download_translation(file_url):
    """
    Download the translated file from the URL

    Args:
        file_url (str): URL to download the translated file from

    Returns:
        str: Content of the translated file, or None if failed
    """
    try:
        # If the URL is not a URL but direct content, return it
        if not file_url.startswith('http'):
            return file_url

        response = requests.get(file_url)
        response.raise_for_status()
        # Explicitly decode as UTF-8 for Hugo markdown content
        return response.content.decode('utf-8')

    except Exception as e:
        print(f"Error downloading translation from {file_url}: {str(e)}")
        return None

def resolve_source_files(only_files, en_dir, content_dir):
    """
    Resolve an explicit list of source files (e.g. the files changed in a PR)
    down to translatable English source files.

    Each entry may be absolute, relative to the current working directory, or
    relative to the content directory (so both `content/en/foo.md` from a repo
    root and `en/foo.md` from the content dir work). Entries that do not exist,
    live outside the English source tree, or are not translatable are skipped
    with a warning.

    Args:
        only_files (list): Raw path strings to resolve
        en_dir (Path): Path to the English source directory (content/en)
        content_dir (Path): Path to the content directory

    Returns:
        list: Existing, de-duplicated, translatable English source Paths
    """
    en_dir_resolved = en_dir.resolve()
    resolved = []
    seen = set()

    for raw in only_files:
        candidate = Path(raw)
        # Build the list of locations to try, in priority order.
        if candidate.is_absolute():
            attempts = [candidate]
        else:
            attempts = [Path.cwd() / candidate, content_dir / candidate]

        chosen = next((p.resolve() for p in attempts if p.exists()), None)
        if chosen is None:
            print(f"  Skipping (not found): {raw}")
            continue

        # Only English source files drive translation; ignore anything else
        # (e.g. a changed file under another language dir).
        try:
            rel = chosen.relative_to(en_dir_resolved)
        except ValueError:
            print(f"  Skipping (not under en/): {raw}")
            continue

        if not is_translatable_file(chosen):
            print(f"  Skipping (not a translatable file type): {raw}")
            continue

        # Rebase onto the (possibly unresolved) en_dir the caller passed in, so
        # downstream `relative_to(en_dir)` stays consistent even when the
        # content path contains a symlink component (e.g. macOS /tmp).
        normalized = en_dir / rel
        if normalized not in seen:
            seen.add(normalized)
            resolved.append(normalized)

    return resolved


def find_files_for_translation(content_dir, target_langs, only_files=None, force=False):
    """
    Find all files that need translation

    Args:
        content_dir (Path): Path to the content directory
        target_langs (list): List of target language codes
        only_files (list): Optional explicit list of English source files to
            translate. When provided, the English tree is NOT walked and only
            these files are considered.
        force (bool): When True, (re)translate even if the target file already
            exists, overwriting it. When False, existing targets are skipped.

    Returns:
        list: List of tuples (file_path, content, target_lang, target_file)
    """
    en_dir = content_dir / "en"
    translation_tasks = []
    files_already_exist = 0

    # Find the translatable English source files: either the explicit --files
    # list, or a full walk of the English directory.
    if only_files:
        translatable_files = resolve_source_files(only_files, en_dir, content_dir)
        print(f"Selected {len(translatable_files)} translatable file(s) from --files")
    else:
        translatable_files = []
        for root, _, files in os.walk(en_dir):
            for file in files:
                file_path = Path(root) / file
                if is_translatable_file(file_path):
                    translatable_files.append(file_path)
        print(f"Found {len(translatable_files)} translatable files in the English directory")

    if len(translatable_files) == 0:
        print("No translatable files found in the English directory")
        return [], 0

    # Create the list of translation tasks
    for file_path in translatable_files:
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Get the relative path from the English directory
        rel_path = file_path.relative_to(en_dir)

        # For each target language, check if translation is needed
        for target_lang in target_langs:
            target_dir = content_dir / target_lang
            target_file = target_dir / rel_path

            # Skip if the target file already exists, unless --force was given.
            if target_file.exists() and not force:
                files_already_exist += 1
                continue

            # Add to translation tasks
            translation_tasks.append((file_path, content, target_lang, target_file))

    return translation_tasks, files_already_exist

def process_translations(translation_tasks, flow_id, workspace_id, max_scheduled_tasks=500):
    """
    Process translation tasks using FlowHunt API with new session-based workflow

    This function creates flow sessions for each translation task, monitors them,
    and downloads the translated files when ready.

    Args:
        translation_tasks (list): List of translation tasks
        flow_id (str): FlowHunt flow ID
        workspace_id (str): FlowHunt workspace ID
        max_scheduled_tasks (int): Maximum number of translation tasks to schedule at once
    """
    if not translation_tasks:
        print("No files need translation (all files already exist in target languages)")
        return

    print(f"Translating {len(translation_tasks)} files with maximum {max_scheduled_tasks} tasks at a time")
    check_interval = 5  # Check every 5 seconds

    # Initialize the API client
    with initialize_api_client() as api_client:
        api_instance = flowhunt.FlowsApi(api_client)

        # Lists to track completed and failed tasks across all batches
        all_completed_tasks = []
        all_failed_tasks = []
        # Frontmatter outcomes, reported in the run summary: pages whose TOML
        # was repaired on the way in, and pages written with frontmatter that
        # still does not parse and therefore need a human.
        repaired_frontmatter = []
        invalid_frontmatter = []

        # Process translations while maintaining max_scheduled_tasks in queue
        remaining_tasks = translation_tasks.copy()
        pending_sessions = {}  # {session_id: (file_path, target_lang, target_file, session_info)}
        completed_tasks = []
        failed_tasks = []
        total_scheduled = 0
        total_completed = 0
        aborted = False

        print(f"\nStarting translation of {len(translation_tasks)} files")
        print(f"Maintaining up to {max_scheduled_tasks} tasks in the queue at all times")

        # Initial progress bar for scheduling
        scheduling_progress = tqdm(total=len(translation_tasks), desc="Scheduling translations")
        processing_progress = tqdm(total=len(translation_tasks), desc="Processing translations")

        # Initial scheduling of tasks up to max_scheduled_tasks
        initial_batch = remaining_tasks[:max_scheduled_tasks]
        remaining_tasks = remaining_tasks[max_scheduled_tasks:]

        # Schedule initial batch of tasks
        for file_path, content, target_lang, target_file in initial_batch:
            session_info = create_translation_session(
                api_instance, file_path, content, target_lang, flow_id, workspace_id
            )

            if session_info:
                # Add to pending sessions
                session_id = session_info['session_id']
                pending_sessions[session_id] = (file_path, target_lang, target_file, session_info)
                total_scheduled += 1
            else:
                # Failed to create session
                failed_tasks.append((file_path, target_lang, target_file))
                all_failed_tasks.append((file_path, target_lang, target_file))

            scheduling_progress.update(1)

        print(f"Initially scheduled {len(pending_sessions)} sessions, now processing and scheduling more as needed...")

        # Continue processing and scheduling until all tasks are completed
        while pending_sessions or remaining_tasks:
            # Wait for the check interval before checking results
            time.sleep(check_interval)

            # Check for completed sessions
            session_ids = list(pending_sessions.keys())
            completed_in_batch = 0
            newly_scheduled = 0

            for session_id in session_ids:
                file_path, target_lang, target_file, session_info = pending_sessions[session_id]

                is_ready, result = check_session_results(api_instance, session_info)

                if is_ready:
                    # Remove from pending sessions
                    del pending_sessions[session_id]
                    completed_in_batch += 1
                    total_completed += 1

                    if result:
                        try:
                            # Download the translated content
                            translated_text = download_translation(result)

                            if translated_text:
                                # Trim all whitespace from the translated text
                                translated_text = translated_text.strip()

                                # Unwrap the markdown code fence the model
                                # sometimes puts around the whole file. This used
                                # to happen inside the `with open(..., 'w')`
                                # block below — i.e. after the target file had
                                # already been truncated — so any error in here
                                # left a 0-byte file behind. It also has to run
                                # before the frontmatter check, or the leading
                                # fence hides the +++ delimiter.
                                if translated_text.startswith("```"):
                                    translated_text = translated_text[3:]
                                if translated_text.endswith("```"):
                                    translated_text = translated_text[:-3]
                                # Also remove markdown code block language markers
                                if translated_text.startswith("markdown\n"):
                                    translated_text = translated_text[9:]
                                translated_text = translated_text.strip()

                                # Validate the TOML frontmatter before writing,
                                # repairing it where the fix is unambiguous. See
                                # validate_toml_frontmatter() for why.
                                frontmatter_error = validate_toml_frontmatter(translated_text)

                                if frontmatter_error:
                                    fixed_text, fixed_lines = repair_toml_frontmatter(translated_text)
                                    if fixed_text:
                                        translated_text = fixed_text
                                        frontmatter_error = None
                                        line_list = ', '.join(str(n) for n in fixed_lines)
                                        print(f"[REPAIRED] {target_file}: invalid TOML frontmatter "
                                              f"fixed on line(s) {line_list}")
                                        repaired_frontmatter.append((target_file, line_list))
                                    else:
                                        # Written anyway, on purpose: a broken
                                        # file can be fixed in the PR, a missing
                                        # one silently leaves the page
                                        # untranslated and needs a whole re-run.
                                        print(f"[INVALID FRONTMATTER] {target_file}: {frontmatter_error}")
                                        print(f"[INVALID FRONTMATTER] writing it anyway — this page "
                                              f"will fail the Hugo build until it is fixed by hand")
                                        invalid_frontmatter.append((target_file, frontmatter_error))

                                # Ensure the target directory exists
                                os.makedirs(target_file.parent, exist_ok=True)

                                # Write the translated content to the target file
                                with open(target_file, 'w', encoding='utf-8') as f:
                                    f.write(translated_text)

                                # Add to completed tasks
                                completed_tasks.append((file_path, target_lang, target_file))
                                all_completed_tasks.append((file_path, target_lang, target_file))
                                print(f"Translated: {target_file}")
                            else:
                                # Failed to download
                                failed_tasks.append((file_path, target_lang, target_file))
                                all_failed_tasks.append((file_path, target_lang, target_file))
                                print(f"Failed to download translation for {file_path} to {target_lang}")

                        except Exception as e:
                            print(f"Error saving translation to {target_file}: {str(e)}")
                            failed_tasks.append((file_path, target_lang, target_file))
                            all_failed_tasks.append((file_path, target_lang, target_file))
                    else:
                        # Translation failed
                        failed_tasks.append((file_path, target_lang, target_file))
                        all_failed_tasks.append((file_path, target_lang, target_file))
                        print(f"Failed to translate {file_path} to {target_lang}")

            # Print batch summary
            if completed_in_batch > 0:
                print(f"[DEBUG] Completed {completed_in_batch} tasks in this batch")

            # Refill the queue up to max_scheduled_tasks.
            #
            # This used to schedule min(completed_in_batch, ...) — one new task
            # per completed one. A task whose session could not be created is
            # never "completed", so every creation failure shrank the queue by
            # one permanently, and once the queue hit zero the loop span forever:
            # nothing pending to complete, therefore nothing scheduled, therefore
            # nothing pending. A run that hit 401 on its first five creations sat
            # at 0/369 for hours until the job timeout killed it.
            #
            # Refilling by free slots instead makes the queue self-healing, and
            # guarantees the loop terminates even when every creation fails.
            free_slots = max(0, max_scheduled_tasks - len(pending_sessions))
            tasks_to_schedule = min(free_slots, len(remaining_tasks))
            if tasks_to_schedule > 0:
                print(f"[DEBUG] Scheduling {tasks_to_schedule} new task(s) to refill the queue")

            for i in range(tasks_to_schedule):
                file_path, content, target_lang, target_file = remaining_tasks.pop(0)
                session_info = create_translation_session(
                    api_instance, file_path, content, target_lang, flow_id, workspace_id
                )

                if session_info:
                    # Add to pending sessions
                    session_id = session_info['session_id']
                    pending_sessions[session_id] = (file_path, target_lang, target_file, session_info)
                    newly_scheduled += 1
                    total_scheduled += 1
                else:
                    # Failed to create session
                    failed_tasks.append((file_path, target_lang, target_file))
                    all_failed_tasks.append((file_path, target_lang, target_file))

                scheduling_progress.update(1)

            # Nothing in flight and not one session could be created this round:
            # the next round will fail identically (bad key, wrong workspace, API
            # down), so stop instead of grinding the whole backlog into the same
            # error and emitting one traceback per file.
            if tasks_to_schedule > 0 and newly_scheduled == 0 and not pending_sessions:
                print(f"[ERROR] Could not create any translation session out of "
                      f"{tasks_to_schedule} attempt(s), and nothing is in flight. Aborting.")
                print("[ERROR] A 401 here usually means FLOWHUNT_API_KEY was issued in a "
                      "different workspace than FLOWHUNT_WORKSPACE_ID — workspace-scoped "
                      "calls return 401 while unscoped ones still succeed.")
                aborted = True
                break

            # Update progress
            processing_progress.update(completed_in_batch)

            # Print status update
            if pending_sessions:
                print(f"[STATUS] Sessions in queue: {len(pending_sessions)} | "
                      f"Completed: {total_completed}/{len(translation_tasks)} | "
                      f"Remaining to schedule: {len(remaining_tasks)} | "
                      f"Just completed: {completed_in_batch} | "
                      f"Just scheduled: {newly_scheduled}")

        # Close the progress bars
        scheduling_progress.close()
        processing_progress.close()

        # Print summary
        print(f"\n[DEBUG] Translation Batch Summary:")
        print(f"[DEBUG] Files translated successfully: {len(all_completed_tasks)}")
        print(f"[DEBUG] Files failed: {len(all_failed_tasks)}")
        print(f"[DEBUG] Total files processed: {len(all_completed_tasks) + len(all_failed_tasks)}")

    # Print overall summary
    print("\n[DEBUG] Overall Translation Summary:")
    print(f"[DEBUG] Files translated successfully: {len(all_completed_tasks)}")
    print(f"[DEBUG] Files failed: {len(all_failed_tasks)}")
    print(f"[DEBUG] Total files processed: {len(all_completed_tasks) + len(all_failed_tasks)}")
    print(f"[DEBUG] Frontmatter repaired: {len(repaired_frontmatter)}")
    print(f"[DEBUG] Frontmatter still invalid: {len(invalid_frontmatter)}")

    if repaired_frontmatter:
        print("\n[DEBUG] Pages whose TOML frontmatter was repaired automatically:")
        for target_file, line_list in repaired_frontmatter:
            print(f"[DEBUG]   {target_file} (line(s) {line_list})")

    if invalid_frontmatter:
        # Loud and last, so it is the final thing in the job log: these pages
        # were written with frontmatter Hugo cannot parse and will fail the
        # per-language build until someone fixes them.
        print(f"\n[ERROR] {len(invalid_frontmatter)} page(s) have invalid TOML frontmatter "
              f"and WILL FAIL the Hugo build:")
        for target_file, error in invalid_frontmatter:
            print(f"[ERROR]   {target_file}")
            print(f"[ERROR]     {error}")
        print("[ERROR] The files were written on purpose so the translation is not lost. "
              "Fix the frontmatter quoting in the PR.")

    print(f"[DEBUG] Translation process completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Exit non-zero so the caller fails. Without this the abort above returned
    # normally, build_content.sh printed "Translation of missing content
    # completed!" and the workflow went green having translated nothing — the
    # exact silent-success the abort was added to prevent.
    if aborted:
        print("[ERROR] Aborted before any translation succeeded — failing the run.")
        sys.exit(1)

def main():
    """Main function to parse arguments and process files"""
    print(f"\n[DEBUG] ========== TRANSLATION SCRIPT STARTING ===========")
    print(f"[DEBUG] Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DEBUG] Script directory: {script_dir}")
    print(f"[DEBUG] Hugo root: {hugo_root}")
    parser = argparse.ArgumentParser(
        description="Translate missing files from English to other languages using FlowHunt API",
        epilog="""
Examples:
  python translate_with_flowhunt.py
  python translate_with_flowhunt.py --path /path/to/content
  python translate_with_flowhunt.py --check-interval 30
  python translate_with_flowhunt.py --flow-id "custom-flow-id"
  python translate_with_flowhunt.py --max-scheduled-tasks 100
  python translate_with_flowhunt.py --files content/en/foo.md content/en/bar.md --force
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Default path is ../content/ relative to the script location
    default_path = os.path.join(hugo_root, "content")
    
    parser.add_argument(
        "--path",
        help="Path to the content directory containing language subdirectories (default: %(default)s)",
        default=default_path
    )
    parser.add_argument(
        "--check-interval",
        help="Interval in seconds to check for completed translation sessions (default: %(default)s)",
        type=int,
        default=5
    )
    parser.add_argument(
        "--max-scheduled-tasks",
        help="Maximum number of scheduled translation tasks (default: %(default)s), once batch is done, next batch will be scheduled",
        type=int,
        default=100
    )
    parser.add_argument(
        "--flow-id",
        help="FlowHunt flow ID for translation service (default: %(default)s)",
        default=DEFAULT_FLOW_ID
    )
    parser.add_argument(
        "--workspace-id",
        help="FlowHunt workspace that owns the flow. Must be the workspace the "
             "API key was issued in — a key from another workspace gets 401 on "
             "every workspace-scoped call (default: %(default)s)",
        default=DEFAULT_WORKSPACE_ID
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit list of English source files to translate (e.g. the files "
             "changed in a PR). Paths may be absolute or relative to the current "
             "directory or the content dir. When given, the English tree is not "
             "walked and only these files are considered."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-translate and overwrite target files even if they already exist "
             "(default: existing translations are skipped)."
    )

    args = parser.parse_args()

    print(f"[DEBUG] Parsed arguments:")
    print(f"[DEBUG] - Path: {args.path}")
    print(f"[DEBUG] - Check interval: {args.check_interval} seconds")
    print(f"[DEBUG] - Max scheduled tasks: {args.max_scheduled_tasks}")
    print(f"[DEBUG] - Flow ID: {args.flow_id}")
    print(f"[DEBUG] - Workspace ID: {args.workspace_id}")
    print(f"[DEBUG] - Files: {args.files if args.files else 'ALL (full en/ walk)'}")
    print(f"[DEBUG] - Force overwrite: {args.force}")
    
    # Convert to Path object
    content_dir = Path(args.path)

    print(f"[DEBUG] Getting workspace ID...")
    workspace_id = get_workspace_id(args.workspace_id)
    if not workspace_id:
        print("[ERROR] Unable to retrieve workspace ID. Please check your API key.")
        sys.exit(1)
    else:
        print(f"[DEBUG] Using workspace ID: {workspace_id}")
    
    # Check if the content directory exists
    print(f"[DEBUG] Checking content directory: {content_dir}")
    if not content_dir.exists() or not content_dir.is_dir():
        print(f"[ERROR] Content directory not found: {content_dir}")
        sys.exit(1)
    print(f"[DEBUG] Content directory exists")
    
    # Check if the English directory exists
    en_dir = content_dir / "en"
    print(f"[DEBUG] Checking English directory: {en_dir}")
    if not en_dir.exists() or not en_dir.is_dir():
        print(f"[ERROR] English directory not found: {en_dir}")
        sys.exit(1)
    print(f"[DEBUG] English directory exists")
    
    # Get target languages
    print(f"[DEBUG] Getting target languages from content directory...")
    target_langs = get_target_languages(content_dir)
    print(f"[DEBUG] Found {len(target_langs)} target languages: {', '.join(target_langs) if target_langs else 'None'}")
    
    if not target_langs:
        print("No target language directories found.")
        sys.exit(0)

    # An explicitly-empty --files list (e.g. a PR that touched no English
    # source files) means "nothing to translate" — never fall back to a full
    # site walk, which would be a surprising and expensive no-scope run.
    if args.files is not None and len(args.files) == 0:
        print("--files was provided but empty; no files to translate. Exiting.")
        sys.exit(0)

    print(f"Content directory: {content_dir}")
    print(f"Source language: en")
    print(f"Target languages: {', '.join(target_langs)}")
    print(f"Using FlowHunt flow ID: {args.flow_id}")

    # Find files that need translation
    print(f"\n[DEBUG] ========== SCANNING FOR FILES TO TRANSLATE ===========")
    translation_tasks, files_already_exist = find_files_for_translation(
        content_dir, target_langs, only_files=args.files, force=args.force
    )
    print(f"[DEBUG] ========== FILE SCAN COMPLETE ===========")
    
    print(f"Found {len(translation_tasks)} files that need translation")
    print(f"Files skipped (already exist): {files_already_exist}")
    
    # Process translations with max-scheduled-tasks parameter
    print(f"\n[DEBUG] ========== STARTING TRANSLATION PROCESS ===========")
    process_translations(translation_tasks, args.flow_id, workspace_id, args.max_scheduled_tasks)
    print(f"[DEBUG] ========== TRANSLATION PROCESS COMPLETE ===========")
    
    print("\n[DEBUG] Translation script completed!")
    print(f"[DEBUG] End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[DEBUG] ========== SCRIPT FINISHED ===========")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[DEBUG] Script interrupted by user (Ctrl+C)")
        print(f"[DEBUG] Interrupted at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        print(f"[DEBUG] Error occurred at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
