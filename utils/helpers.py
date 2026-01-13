"""
Common utility functions for the Reconciliation App.
"""
import logging
from datetime import datetime
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_bytes(size: int) -> str:
    """
    Convert bytes to human-readable format.
    
    Args:
        size: Size in bytes
        
    Returns:
        Human-readable size string (e.g., '1.5 MB')
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def format_number(num: int) -> str:
    """
    Format number with comma separators.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string (e.g., '1,234,567')
    """
    return f"{num:,}"


def get_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        ISO formatted timestamp string
    """
    return datetime.now().isoformat()


def safe_get(data: dict, key: str, default: Any = None) -> Any:
    """
    Safely get a value from a dictionary.
    
    Args:
        data: Dictionary to search
        key: Key to find
        default: Default value if key not found
        
    Returns:
        Value if found, else default
    """
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        return default


def validate_column_name(name: str) -> bool:
    """
    Validate that a column name is non-empty and valid.
    
    Args:
        name: Column name to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not name or not isinstance(name, str):
        return False
    return len(name.strip()) > 0


def clean_column_names(columns: list) -> list:
    """
    Clean and standardize column names.
    
    Args:
        columns: List of column names
        
    Returns:
        List of cleaned column names
    """
    cleaned = []
    for col in columns:
        # Strip whitespace and convert to string
        clean_col = str(col).strip()
        # Replace special characters with underscores
        clean_col = clean_col.replace(' ', '_').replace('-', '_')
        cleaned.append(clean_col)
    return cleaned
