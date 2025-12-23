"""
Robust TOML frontmatter parser using tomllib (Python 3.11+ stdlib).

The python-frontmatter library uses the 'toml' package which has bugs with
certain multi-line strings and outputs booleans as True/False instead of true/false.
This module provides a drop-in replacement that uses tomllib for reliable parsing
and line-based updates for safe writing.
"""

import re
import tomllib
from typing import Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class Post:
    """Simple Post class compatible with python-frontmatter."""
    content: str = ""
    metadata: dict = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.metadata[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.metadata


def parse_toml_frontmatter(content: str) -> Tuple[dict, str]:
    """
    Parse TOML frontmatter from Hugo markdown content.

    Args:
        content: Full file content with +++ delimited frontmatter

    Returns:
        Tuple of (metadata dict, body content)
    """
    # Match TOML frontmatter (between +++ markers)
    match = re.match(r'^\+\+\+\s*\n(.*?)\n\+\+\+\s*\n?(.*)', content, re.DOTALL)

    if not match:
        # No TOML frontmatter found
        return {}, content

    fm_content = match.group(1)
    body = match.group(2)

    try:
        metadata = tomllib.loads(fm_content)
        return metadata, body
    except tomllib.TOMLDecodeError:
        # Return empty metadata on parse failure
        return {}, content


def loads(content: str, handler=None) -> Post:
    """
    Load frontmatter from string content.

    Compatible with frontmatter.loads() API.

    Args:
        content: Full file content
        handler: Ignored (for compatibility)

    Returns:
        Post object with metadata and content
    """
    metadata, body = parse_toml_frontmatter(content)
    return Post(content=body, metadata=metadata)


def load(fp, handler=None) -> Post:
    """
    Load frontmatter from file object.

    Compatible with frontmatter.load() API.
    """
    content = fp.read()
    return loads(content, handler=handler)


def _format_toml_value(value: Any) -> str:
    """Format a Python value as a TOML value string."""
    if isinstance(value, str):
        # Escape quotes in strings
        escaped = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{escaped}"'
    elif isinstance(value, bool):
        return str(value).lower()  # true/false, not True/False
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, dict):
        # Format as TOML inline table
        items = ', '.join(f'{k} = {_format_toml_value(v)}' for k, v in value.items())
        return f'{{ {items} }}'
    elif isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            items = ', '.join(f'"{v}"' for v in value)
            return f'[ {items} ]'
        else:
            items = ', '.join(_format_toml_value(v) for v in value)
            return f'[ {items} ]'
    elif value is None:
        return '""'  # Empty string for None
    else:
        return str(value)


def dumps(post: Post, handler=None) -> str:
    """
    Serialize Post back to frontmatter format.

    WARNING: This rewrites the entire frontmatter. For updating single attributes,
    use update_frontmatter_attribute() instead to preserve formatting.

    Args:
        post: Post object to serialize
        handler: Ignored (for compatibility)

    Returns:
        String with TOML frontmatter
    """
    lines = ["+++"]

    for key, value in post.metadata.items():
        if isinstance(value, str):
            # Use multi-line string for long content or content with newlines
            if '\n' in value or len(value) > 100:
                lines.append(f'{key} = """')
                lines.append(value)
                lines.append('"""')
            else:
                escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                lines.append(f'{key} = "{escaped}"')
        elif isinstance(value, bool):
            lines.append(f'{key} = {str(value).lower()}')
        elif isinstance(value, (int, float)):
            lines.append(f'{key} = {value}')
        elif isinstance(value, list):
            if all(isinstance(v, str) for v in value):
                items = ', '.join(f'"{v}"' for v in value)
                lines.append(f'{key} = [ {items},]')
            elif all(isinstance(v, dict) for v in value):
                # Array of tables
                for item in value:
                    lines.append(f'[[{key}]]')
                    for k, v in item.items():
                        if isinstance(v, str):
                            escaped = v.replace('\\', '\\\\').replace('"', '\\"')
                            if '\n' in v:
                                lines.append(f'{k} = """')
                                lines.append(v)
                                lines.append('"""')
                            else:
                                lines.append(f'{k} = "{escaped}"')
                        elif isinstance(v, bool):
                            lines.append(f'{k} = {str(v).lower()}')
                        else:
                            lines.append(f'{k} = {v}')
                    lines.append('')
            else:
                items = ', '.join(str(v) for v in value)
                lines.append(f'{key} = [ {items} ]')
        elif value is None:
            continue  # Skip None values
        else:
            lines.append(f'{key} = {value}')

    lines.append("+++")
    lines.append("")
    lines.append(post.content)

    return '\n'.join(lines)


# Compatibility: provide a TOMLHandler-like interface
class TOMLHandler:
    """Compatibility class for frontmatter.TOMLHandler."""

    START_DELIMITER = "+++"
    END_DELIMITER = "+++"
    FM_BOUNDARY = re.compile(r'^\+\+\+\s*$', re.MULTILINE)

    def detect(self, text: str) -> bool:
        return text.startswith('+++')

    def load(self, fm: str) -> dict:
        try:
            return tomllib.loads(fm)
        except tomllib.TOMLDecodeError:
            return {}

    def export(self, metadata: dict) -> str:
        lines = []
        for key, value in metadata.items():
            if isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, bool):
                lines.append(f'{key} = {str(value).lower()}')
            elif isinstance(value, list):
                items = ', '.join(f'"{v}"' if isinstance(v, str) else str(v) for v in value)
                lines.append(f'{key} = [ {items} ]')
            else:
                lines.append(f'{key} = {value}')
        return '\n'.join(lines)


def update_frontmatter_attribute(content: str, key: str, value: Any) -> str:
    """
    Safely update a single attribute in TOML frontmatter using line-based approach.

    This function ONLY modifies the specified attribute line, preserving all other
    content including nested tables, booleans, and complex structures.

    Args:
        content: Full file content with +++ delimited frontmatter
        key: Attribute name to update (e.g., 'keywords', 'linkbuilding')
        value: New value (string, list of strings, bool, int, etc.)

    Returns:
        Updated content with the attribute modified
    """
    if not content.startswith('+++'):
        return content

    match = re.match(r'^(\+\+\+\s*\n)(.*?)(\n\+\+\+\s*\n?)(.*)', content, re.DOTALL)
    if not match:
        return content

    opening, raw_fm, closing, body = match.groups()

    # Format the value for TOML
    if isinstance(value, list):
        # List of strings
        items = ', '.join(f'"{v}"' for v in value)
        value_str = f'[ {items} ]'
    elif isinstance(value, bool):
        value_str = str(value).lower()
    elif isinstance(value, str):
        value_str = f'"{value}"'
    else:
        value_str = str(value)

    new_line = f'{key} = {value_str}'

    # Process lines
    lines = raw_fm.split('\n')
    filtered_lines = []
    key_found = False
    title_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Check if this line is the attribute we're updating
        if stripped.startswith(f'{key}') and '=' in stripped:
            # Check it's exactly the key (not a prefix match)
            key_part = stripped.split('=')[0].strip()
            if key_part == key:
                key_found = True
                continue  # Skip this line, we'll add the new one
        filtered_lines.append(line)
        if stripped.startswith('title') and '=' in stripped:
            title_idx = len(filtered_lines)

    # Insert the new attribute line
    if key_found:
        # Insert at the position where we removed it (after title if possible)
        insert_idx = title_idx if title_idx > 0 else 1
    else:
        # Add new attribute after title
        insert_idx = title_idx if title_idx > 0 else 1

    filtered_lines.insert(insert_idx, new_line)

    new_fm = '\n'.join(filtered_lines)
    return f"{opening}{new_fm}{closing}{body}"


def safe_toml_dumps(data: dict) -> str:
    """
    Serialize a dictionary to TOML string with correct boolean formatting.

    Unlike the 'toml' library, this outputs lowercase 'true'/'false' for booleans.

    Args:
        data: Dictionary to serialize

    Returns:
        TOML-formatted string
    """
    lines = []

    # Separate simple values from arrays of tables
    simple_values = {}
    array_tables = {}
    nested_tables = {}

    for key, value in data.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            array_tables[key] = value
        elif isinstance(value, dict):
            nested_tables[key] = value
        else:
            simple_values[key] = value

    # Output simple values first
    for key, value in simple_values.items():
        lines.append(f'{key} = {_format_toml_value(value)}')

    # Output nested tables
    for table_name, table_data in nested_tables.items():
        if lines:
            lines.append('')
        lines.append(f'[{table_name}]')
        for key, value in table_data.items():
            if isinstance(value, dict):
                # Sub-table
                lines.append('')
                lines.append(f'[{table_name}.{key}]')
                for k, v in value.items():
                    lines.append(f'{k} = {_format_toml_value(v)}')
            else:
                lines.append(f'{key} = {_format_toml_value(value)}')

    # Output array of tables
    for table_name, items in array_tables.items():
        for item in items:
            if lines:
                lines.append('')
            lines.append(f'[[{table_name}]]')

            # Separate simple values from nested in array items
            simple_item_values = {}
            nested_item_values = {}

            for key, value in item.items():
                if isinstance(value, dict):
                    nested_item_values[key] = value
                else:
                    simple_item_values[key] = value

            for key, value in simple_item_values.items():
                lines.append(f'{key} = {_format_toml_value(value)}')

            for nested_key, nested_value in nested_item_values.items():
                lines.append('')
                lines.append(f'[{table_name}.{nested_key}]')
                for k, v in nested_value.items():
                    lines.append(f'{k} = {_format_toml_value(v)}')

    return '\n'.join(lines)
