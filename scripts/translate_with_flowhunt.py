#!/usr/bin/env python3
"""
translate_with_flowhunt.py

This script translates files from /content/en/* to all language variations defined in /content/[lang]/
that don't already exist in the target language directories using FlowHunt API.

Usage:
    python translate_with_flowhunt.py [--path /path/to/content] [--check-interval 60] [--flow-id FLOW_ID] [--max-scheduled-tasks LIMIT]

Prerequisites:
    - Python 3.6 or higher
    - FlowHunt API key (set in .env file or as environment variable API_KEY)
    - Required packages: flowhunt, tqdm, python-dotenv

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
import sys
import argparse
import time
import json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import flowhunt
from pprint import pprint

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

# Default FlowHunt flow ID and workspace ID for translation service
DEFAULT_FLOW_ID = '7389730a-fbaf-48a2-bb77-3b6814c23b20'

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



def get_workspace_id():
    api_client = initialize_api_client()
    # Create an instance of the API class
    api_instance = flowhunt.WebAuthApi(api_client)

    try:
        # Get User
        api_response = api_instance.get_user()
        return api_response.api_key_workspace_id
    except flowhunt.ApiException as e:
        print("Exception when calling WebAuthApi->get_user: %s\n" % e)
        return None
    

def is_translatable_file(file_path):
    """Check if a file should be translated based on extension"""
    return file_path.suffix.lower() in ['.md', '.markdown', '.yaml', '.yml', '.html', '.txt']

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

def invoke_flow_for_translation(api_instance, content, target_lang, flow_id, workspace_id):
    print(f"[DEBUG] Invoking flow for {target_lang} translation...")
    """
    Invoke a FlowHunt flow to translate content to the target language
    
    Args:
        api_instance: FlowHunt API instance
        content (str): Content to translate
        target_lang (str): Target language code
        flow_id (str): FlowHunt flow ID
        workspace_id (str): FlowHunt workspace ID
        
    Returns:
        str: Process ID or None if failed
    """
    try:
        # Get the full language name from the map, fallback to the code if not found
        language_name = LANGUAGE_MAP.get(target_lang.lower(), target_lang)
        
        # Prepare the request payload
        flow_invoke_request = flowhunt.FlowInvokeRequest(
            variables={
                "source_language": "English",
                "target_language": language_name,
                "today": time.strftime("%Y-%m-%d %H:00:00"),
            }, 
            human_input=content
        )
        
        # Invoke the flow
        response = api_instance.invoke_flow_singleton(
            flow_id=flow_id,
            workspace_id=workspace_id,
            flow_invoke_request=flow_invoke_request
        )
        
        # Return the process ID for checking status later
        return response.id
        
    except Exception as e:
        print(f"Error invoking flow for {target_lang}: {str(e)}")
        return None

def check_flow_results(api_instance, process_id, flow_id, workspace_id):
    # Note: This function is called frequently, so we'll skip detailed debug messages here to avoid spam
    """
    Check if a flow has completed and get the results
    
    Args:
        api_instance: FlowHunt API instance
        process_id (str): Process ID to check
        flow_id (str): FlowHunt flow ID
        workspace_id (str): FlowHunt workspace ID
        
    Returns:
        tuple: (is_ready, result_text)
    """
    try:
        # Get the results of the invoked flow
        response = api_instance.get_invoked_flow_results(
            flow_id=flow_id, task_id=process_id, workspace_id=workspace_id
        )
        
        # Check if the flow has completed
        if response.status == "SUCCESS":
            # Extract the translated text from the response
            translated_text = json.loads(response.result)
            translated_text = translated_text['outputs'][0]['outputs'][0]['results']['message']['result']
            return True, translated_text
        else:
            # Flow is still processing
            return False, None
            
    except Exception as e:
        print(f"Error checking flow results for process {process_id}: {str(e)}")
        return False, None

def find_files_for_translation(content_dir, target_langs):
    """
    Find all files that need translation
    
    Args:
        content_dir (Path): Path to the content directory
        target_langs (list): List of target language codes
        
    Returns:
        list: List of tuples (file_path, content, target_lang, target_file)
    """
    en_dir = content_dir / "en"
    translation_tasks = []
    files_already_exist = 0
    
    # Find all translatable files in the English directory
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
            
            # Skip if the target file already exists
            if target_file.exists():
                files_already_exist += 1
                continue
            
            # Add to translation tasks
            translation_tasks.append((file_path, content, target_lang, target_file))
    
    return translation_tasks, files_already_exist

def process_translations(translation_tasks, flow_id, workspace_id, max_scheduled_tasks=500):
    """
    Process translation tasks using FlowHunt API, maintaining a constant queue size
    
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
    check_interval = 10
    # Initialize the API client
    with initialize_api_client() as api_client:
        api_instance = flowhunt.FlowsApi(api_client)
        
        # Lists to track completed and failed tasks across all batches
        all_completed_tasks = []
        all_failed_tasks = []
        
        # Process translations while maintaining max_scheduled_tasks in queue
        remaining_tasks = translation_tasks.copy()
        pending_tasks = {}  # {process_id: (file_path, target_lang, target_file, content)}
        completed_tasks = []
        failed_tasks = []
        total_scheduled = 0
        total_completed = 0
        
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
            process_id = invoke_flow_for_translation(api_instance, content, target_lang, flow_id, workspace_id)
            
            if process_id:
                # Add to pending tasks
                pending_tasks[process_id] = (file_path, target_lang, target_file, content)
                total_scheduled += 1
            else:
                # Failed to invoke flow
                failed_tasks.append((file_path, target_lang, target_file))
                all_failed_tasks.append((file_path, target_lang, target_file))
            
            scheduling_progress.update(1)
        
        print(f"Initially scheduled {len(pending_tasks)} tasks, now processing and scheduling more as needed...")
        
        # Continue processing and scheduling until all tasks are completed
        while pending_tasks or remaining_tasks:
            # Wait for the check interval before checking results
            time.sleep(check_interval)
            
            # Check for completed tasks
            process_ids = list(pending_tasks.keys())
            completed_in_batch = 0
            newly_scheduled = 0
            
            for process_id in process_ids:
                file_path, target_lang, target_file, content = pending_tasks[process_id]
                
                is_ready, translated_text = check_flow_results(api_instance, process_id, flow_id, workspace_id)
                
                # Trim all whitespace from the translated text
                if translated_text:
                    translated_text = translated_text.strip()

                if is_ready:
                    # Remove from pending tasks
                    del pending_tasks[process_id]
                    completed_in_batch += 1
                    total_completed += 1
                    
                    if translated_text:
                        try:
                            # Ensure the target directory exists
                            os.makedirs(target_file.parent, exist_ok=True)
                            
                            # Write the translated content to the target file
                            with open(target_file, 'w', encoding='utf-8') as f:
                                # If translated text starts or ends with ```, remove it
                                if translated_text.startswith("```"):
                                    translated_text = translated_text[3:]
                                if translated_text.endswith("```"):
                                    translated_text = translated_text[:-3]
                                f.write(translated_text)
                            
                            # Add to completed tasks
                            completed_tasks.append((file_path, target_lang, target_file))
                            all_completed_tasks.append((file_path, target_lang, target_file))
                            print(f"Translated: {target_file}")
                            
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
            
            # Schedule new tasks to replace completed ones, maintaining max_scheduled_tasks
            tasks_to_schedule = min(completed_in_batch, len(remaining_tasks))
            if tasks_to_schedule > 0:
                print(f"[DEBUG] Scheduling {tasks_to_schedule} new tasks to replace completed ones")
            
            for i in range(tasks_to_schedule):
                file_path, content, target_lang, target_file = remaining_tasks.pop(0)
                process_id = invoke_flow_for_translation(api_instance, content, target_lang, flow_id, workspace_id)
                
                if process_id:
                    # Add to pending tasks
                    pending_tasks[process_id] = (file_path, target_lang, target_file, content)
                    newly_scheduled += 1
                    total_scheduled += 1
                else:
                    # Failed to invoke flow
                    failed_tasks.append((file_path, target_lang, target_file))
                    all_failed_tasks.append((file_path, target_lang, target_file))
                
                scheduling_progress.update(1)
            
            # Update progress
            processing_progress.update(completed_in_batch)
            
            # Print status update
            if pending_tasks:
                print(f"[STATUS] Tasks in queue: {len(pending_tasks)} | "
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
    print(f"[DEBUG] Translation process completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

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
        help="Interval in seconds to check for completed translation tasks (default: %(default)s)",
        type=int,
        default=60
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
    
    args = parser.parse_args()
    
    print(f"[DEBUG] Parsed arguments:")
    print(f"[DEBUG] - Path: {args.path}")
    print(f"[DEBUG] - Check interval: {args.check_interval} seconds")
    print(f"[DEBUG] - Max scheduled tasks: {args.max_scheduled_tasks}")
    print(f"[DEBUG] - Flow ID: {args.flow_id}")
    
    # Convert to Path object
    content_dir = Path(args.path)

    print(f"[DEBUG] Getting workspace ID...")
    workspace_id = get_workspace_id()
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
    
    print(f"Content directory: {content_dir}")
    print(f"Source language: en")
    print(f"Target languages: {', '.join(target_langs)}")
    print(f"Using FlowHunt flow ID: {args.flow_id}")
    
    # Find files that need translation
    print(f"\n[DEBUG] ========== SCANNING FOR FILES TO TRANSLATE ===========")
    translation_tasks, files_already_exist = find_files_for_translation(content_dir, target_langs)
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
