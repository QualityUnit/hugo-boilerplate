#!/usr/bin/env python3
"""
generate_content.py

This script generates content for each topic provided in a CSV input file using FlowHunt API.

Usage:
    python generate_content.py --input_file topics.csv --flow_id FLOW_ID --output_dir generated_content

Prerequisites:
    - Python 3.6 or higher
    - FlowHunt API key (set in .env file or as environment variable FLOWHUNT_API_KEY)
    - Required packages: flowhunt, tqdm, python-dotenv

CSV Input Format:
    The input CSV file should have the following columns:
    - flow_input: Text used to create request in FlowHunt
    - filename: Filename to write in output directory

Examples:
    # Basic usage
    python generate_content.py --input_file topics.csv --flow_id "flow-id" --output_dir output
    python generate_content.py --input_file glossaries.csv --flow_id 8eeb2771-10c0-4165-a16b-37fe81707659 --output_dir ../../../content/en/glossary
    
    # Prevent overwriting existing files
    python generate_content.py --input_file topics.csv --flow_id "flow-id" --output_dir output --no-overwrite
"""

import os
import sys
import argparse
import time
import json
import csv
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
import flowhunt

# Load environment variables from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
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

def get_workspace_id():
    """Get the workspace ID from FlowHunt API"""
    api_client = initialize_api_client()
    api_instance = flowhunt.AuthApi(api_client)

    try:
        api_response = api_instance.get_user()
        return api_response.api_key_workspace_id
    except flowhunt.ApiException as e:
        print(f"Exception when calling AuthApi->get_user: {e}")
        return None

def initialize_api_client():
    """Initialize and return a FlowHunt API client"""
    configuration = flowhunt.Configuration(
        host="https://api.flowhunt.io"
    )
    configuration.api_key['APIKeyHeader'] = api_key
    return flowhunt.ApiClient(configuration)

def invoke_flow_for_content(api_instance, flow_input, flow_id, workspace_id, filename):
    """
    Invoke a FlowHunt flow to generate content for a flow input
    
    Args:
        api_instance: FlowHunt API instance
        flow_input (str): Input text to generate content for
        flow_id (str): FlowHunt flow ID
        workspace_id (str): FlowHunt workspace ID
        filename (str): Filename to be passed to the flow

    Returns:
        str: Process ID or None if failed
    """
    try:
        # Prepare the request payload
        flow_invoke_request = flowhunt.FlowInvokeRequest(
            variables={
                "today": time.strftime("%Y-%m-%d"),
                "v": "1",
                "filename": filename,
            }, 
            human_input=flow_input
        )
        
        # Invoke the flow
        response = api_instance.invoke_flow_singleton(
            flow_id=flow_id,
            workspace_id=workspace_id,
            flow_invoke_request=flow_invoke_request
        )
        
        return response.id
        
    except Exception as e:
        print(f"Error invoking flow for input '{flow_input}': {str(e)}")
        return None

def check_flow_results(api_instance, process_id, flow_id, workspace_id):
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
        response = api_instance.get_invoked_flow_results(
            flow_id=flow_id, task_id=process_id, workspace_id=workspace_id
        )
        
        if response.status == "SUCCESS":
            generated_content = json.loads(response.result)
            content = ""
            for output in generated_content['outputs'][0]['outputs']:
                part = output['results']['message']['result'].strip()
                if part.startswith("```"):
                    part = "\n".join(part.splitlines()[1:]).strip()
                if part.endswith("```"):
                    part = part[:-3].strip()
                content += part + "\n"
            content = content.strip()

            if not content:
                content = "NOCONTENT"

            return True, content.strip()
        else:
            return False, None
            
    except Exception as e:
        print(f"Error checking flow results for process {process_id}: {str(e)}")
        return False, None

def detect_csv_delimiter(file_path):
    """
    Detect delimiter in a CSV file based on the first line.
    Checks for comma, semicolon, or tab and uses the most frequent one.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        str: Detected delimiter (either ',', ';', or '\t')
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

            # Count occurrences of each potential delimiter
            comma_count = first_line.count(',')
            semicolon_count = first_line.count(';')
            tab_count = first_line.count('\t')

            # Find the most common delimiter
            delimiter_counts = {
                ',': comma_count,
                ';': semicolon_count,
                '\t': tab_count
            }

            max_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            max_count = delimiter_counts[max_delimiter]

            # If none of the delimiters are found, default to semicolon
            if max_count == 0:
                print("No common delimiter found in the first line. Defaulting to semicolon.")
                return ';'

            delimiter_name = {',': 'comma', ';': 'semicolon', '\t': 'tab'}
            print(f"Detected delimiter: '{max_delimiter}' ({delimiter_name.get(max_delimiter, 'unknown')})")
            return max_delimiter
    except Exception as e:
        print(f"Error detecting CSV delimiter: {str(e)}")
        print("Defaulting to semicolon delimiter")
        return ';'

def read_topics(input_file):
    """Read topics from CSV input file"""
    try:
        # Increase CSV field size limit to handle large fields
        csv.field_size_limit(1000000)  # Set to 1MB limit
        
        topics = []
        file_extension = os.path.splitext(input_file)[1].lower()
        
        if file_extension == '.csv':
            # Detect the delimiter in the CSV file
            delimiter = detect_csv_delimiter(input_file)

            # Read CSV file with flow_input and filename columns
            with open(input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                # Validate required columns
                if 'flow_input' not in reader.fieldnames or 'filename' not in reader.fieldnames:
                    print("Error: CSV file must contain 'flow_input' and 'filename' columns")
                    sys.exit(1)
                
                for row in reader:
                    flow_input = row['flow_input'].strip()
                    filename = row['filename'].strip()
                    
                    if flow_input and filename:
                        topics.append({
                            'flow_input': flow_input,
                            'filename': filename
                        })
        else:
            # Legacy support for text files (backwards compatibility)
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # For text files, create filename from topic
                        safe_filename = "".join(x for x in line if x.isalnum() or x in (' ', '-', '_')).strip()[:50]
                        safe_filename = safe_filename.replace(' ', '_') + '.md'
                        topics.append({
                            'flow_input': line,
                            'filename': safe_filename
                        })
        
        return topics
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        sys.exit(1)

def save_content(content, filename, output_dir, allow_overwrite=True):
    """Save generated content to a file"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Ensure output_dir has / at the end
        if not output_dir.endswith('/'):
            output_dir += '/'
        # ensure filename doesn't start with /
        filename = filename.lstrip('/')
        
        output_path = os.path.join(output_dir, filename)
        
        # Create subdirectories if they exist in the filename path
        file_dir = os.path.dirname(output_path)
        if file_dir and file_dir != output_dir:
            os.makedirs(file_dir, exist_ok=True)
        
        # Check if file exists and overwrite is not allowed
        if not allow_overwrite and os.path.exists(output_path):
            print(f"File {output_path} already exists and overwrite is disabled. Skipping.")
            return None
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return output_path
    except Exception as e:
        print(f"Error saving content to '{filename}': {str(e)}")
        return None

def process_topics(topics, flow_id, workspace_id, output_dir, allow_overwrite=True):
    """Process topics using FlowHunt API"""
    if not topics:
        print("No topics to process")
        return
    
    skipped_count = 0
    original_count = len(topics)
    
    # Filter out existing files if overwrite is not allowed
    if not allow_overwrite:
        existing_files = []
        topics_to_process = []
        
        for topic in topics:
            output_path = os.path.join(output_dir, topic['filename'])
            if os.path.exists(output_path):
                existing_files.append(topic['filename'])
                print(f"Skipping '{topic['flow_input']}' - file '{topic['filename']}' already exists")
            else:
                topics_to_process.append(topic)
        
        skipped_count = len(existing_files)
        
        if existing_files:
            print(f"\nSkipped {len(existing_files)} files that already exist:")
            for filename in existing_files:
                print(f"  - {filename}")
            print()
        
        topics = topics_to_process
        
        if not topics:
            print("No new topics to process - all files already exist")
            print(f"\nContent Generation Summary:")
            print(f"Topics skipped (files already exist): {skipped_count}")
            print(f"Topics completed successfully: 0")
            print(f"Topics failed: 0")
            print(f"Total topics in input: {original_count}")
            return
    
    print(f"Processing {len(topics)} topics")
    check_interval = 10
    
    with initialize_api_client() as api_client:
        api_instance = flowhunt.FlowsApi(api_client)
        
        # Dictionary to track tasks: {process_id: topic_data}
        pending_tasks = {}
        completed_tasks = []
        failed_tasks = []
        
        # Schedule all tasks
        progress_bar = tqdm(total=len(topics), desc="Scheduling content generation")
        
        for topic in topics:
            flow_input = topic['flow_input'].strip()
            filename = topic['filename'].strip()
            
            if not flow_input or not filename:
                continue
                
            process_id = invoke_flow_for_content(api_instance, flow_input, flow_id, workspace_id, filename)

            if process_id:
                pending_tasks[process_id] = topic
            else:
                failed_tasks.append(topic)
                progress_bar.update(1)
        
        progress_bar.close()
        print(f"Scheduled {len(pending_tasks)} tasks, now waiting for results...")
        
        # Process results
        progress_bar = tqdm(total=len(pending_tasks), desc="Processing content generation")
        
        while pending_tasks:
            time.sleep(check_interval)
            process_ids = list(pending_tasks.keys())
            completed_in_batch = 0
            
            for process_id in process_ids:
                topic_data = pending_tasks[process_id]
                is_ready, content = check_flow_results(api_instance, process_id, flow_id, workspace_id)
                
                if is_ready:
                    del pending_tasks[process_id]
                    completed_in_batch += 1
                    
                    if content:
                        output_path = save_content(content, topic_data['filename'], output_dir, allow_overwrite)
                        if output_path:
                            completed_tasks.append(topic_data)
                            print(f"\nGenerated content for '{topic_data['flow_input']}' saved to: {output_path}")
                        else:
                            failed_tasks.append(topic_data)
                    else:
                        failed_tasks.append(topic_data)
                        print(f"\nFailed to generate content for '{topic_data['flow_input']}'")
            
            progress_bar.update(completed_in_batch)
            
            if pending_tasks:
                print(f"\nStill waiting for {len(pending_tasks)} tasks to complete...")
        
        progress_bar.close()
    
    # Print summary
    print("\nContent Generation Summary:")
    if skipped_count > 0:
        print(f"Topics skipped (files already exist): {skipped_count}")
    print(f"Topics completed successfully: {len(completed_tasks)}")
    print(f"Topics failed: {len(failed_tasks)}")
    print(f"Total topics in input: {original_count}")
    print(f"Total topics processed: {len(completed_tasks) + len(failed_tasks)}")

def main():
    """Main function to parse arguments and process topics"""
    parser = argparse.ArgumentParser(
        description="Generate content for topics using FlowHunt API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to CSV input file containing flow_input and filename columns"
    )
    parser.add_argument(
        "--flow_id",
        required=True,
        help="FlowHunt flow ID for content generation"
    )
    parser.add_argument(
        "--output_dir",
        required=True,  
        help="Directory where generated content will be saved"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Prevent overwriting existing files (default: allow overwrite)"
    )
    
    args = parser.parse_args()
    
    # Get workspace ID
    workspace_id = get_workspace_id()
    if not workspace_id:
        print("Error: Unable to retrieve workspace ID. Please check your API key.")
        sys.exit(1)
    
    print(f"Using workspace ID: {workspace_id}")
    
    # Read topics
    topics = read_topics(args.input_file)
    print(f"Found {len(topics)} topics in {args.input_file}")
    
    # Process topics
    process_topics(topics, args.flow_id, workspace_id, args.output_dir, allow_overwrite=not args.no_overwrite)
    
    print("\nContent generation completed!")

if __name__ == "__main__":
    main()