"""
File Handler Module for the Reconciliation App.
Handles file upload, validation, parsing, and statistics.
"""
import io
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass
class FileStats:
    """Container for file statistics."""
    rows: int
    columns: int
    size_bytes: int
    column_names: List[str]
    dtypes: Dict[str, str]


class FileHandler:
    """
    Handles file upload, validation, and parsing operations.
    Supports CSV and Excel files using PyArrow for performance.
    """
    
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
    KEY_COLUMN_PATTERNS = [
        'transaction_id', 'txn_id', 'trans_id', 'transaction id',
        'order_id', 'order id', 'orderid',
        'settlement_id', 'settle_id', 'settlement id',
        'reference', 'ref', 'ref_id', 'reference_id',
        'invoice', 'invoice_id', 'invoice_no',
        'id', 'uuid', 'sku', 'product_id'
    ]
    
    def __init__(self, chunk_size: int = 10000):
        """
        Initialize FileHandler.
        
        Args:
            chunk_size: Number of rows to process per chunk for large files
        """
        self.chunk_size = chunk_size
    
    def load_file(
        self, 
        file: Any, 
        file_name: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load a file (CSV or Excel) into a Pandas DataFrame.
        
        Args:
            file: File-like object from Streamlit uploader
            file_name: Original filename to determine type
            
        Returns:
            Tuple of (DataFrame, error_message). Error is None on success.
        """
        try:
            file_extension = self._get_extension(file_name)
            
            if file_extension not in self.SUPPORTED_EXTENSIONS:
                return None, f"Unsupported file type: {file_extension}"
            
            # Read file content
            file_content = file.read()
            file.seek(0)  # Reset for potential re-read
            
            if file_extension == '.csv':
                df = self._load_csv(file_content)
            else:  # Excel files
                df = self._load_excel(file_content)
            
            logger.info(f"Loaded file {file_name}: {len(df)} rows, {len(df.columns)} columns")
            return df, None
            
        except Exception as e:
            logger.error(f"Error loading file {file_name}: {str(e)}")
            return None, f"Error loading file: {str(e)}"
    
    def _load_csv(self, content: bytes) -> pd.DataFrame:
        """
        Load CSV file using PyArrow for performance.
        
        Args:
            content: File content as bytes
            
        Returns:
            Pandas DataFrame
        """
        try:
            # Try PyArrow first for performance
            table = pv.read_csv(io.BytesIO(content))
            return table.to_pandas()
        except Exception:
            # Fallback to Pandas for edge cases
            return pd.read_csv(io.BytesIO(content))
    
    def _load_excel(self, content: bytes) -> pd.DataFrame:
        """
        Load Excel file using Pandas with openpyxl.
        
        Args:
            content: File content as bytes
            
        Returns:
            Pandas DataFrame
        """
        return pd.read_excel(io.BytesIO(content), engine='openpyxl')
    
    def validate_file(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate that a DataFrame is non-empty and has valid structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None:
            return False, "DataFrame is None"
        
        if df.empty:
            return False, "File is empty (no data rows)"
        
        if len(df.columns) == 0:
            return False, "File has no columns"
        
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            return False, "File has duplicate column names"
        
        return True, None
    
    def get_file_stats(self, df: pd.DataFrame, file_size: int) -> FileStats:
        """
        Get statistics for a loaded DataFrame.
        
        Args:
            df: DataFrame to analyze
            file_size: Original file size in bytes
            
        Returns:
            FileStats object with row/column counts and metadata
        """
        return FileStats(
            rows=len(df),
            columns=len(df.columns),
            size_bytes=file_size,
            column_names=list(df.columns),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()}
        )
    
    def get_preview(self, df: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
        """
        Get a preview of the first N rows.
        
        Args:
            df: DataFrame to preview
            rows: Number of rows to show
            
        Returns:
            DataFrame with first N rows
        """
        return df.head(rows)
    
    def detect_key_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Auto-detect potential key/ID columns for joining.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            List of column names that might be join keys
        """
        potential_keys = []
        
        for col in df.columns:
            col_lower = str(col).lower().strip()
            
            # Check against known patterns
            for pattern in self.KEY_COLUMN_PATTERNS:
                if pattern in col_lower or col_lower in pattern:
                    potential_keys.append(col)
                    break
            else:
                # Check if column has unique or near-unique values
                if len(df) > 0:
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio > 0.9:  # 90% unique values
                        potential_keys.append(col)
        
        return list(set(potential_keys))
    
    def _get_extension(self, filename: str) -> str:
        """Get lowercase file extension."""
        if '.' in filename:
            return '.' + filename.rsplit('.', 1)[1].lower()
        return ''
    
    def load_file_chunked(
        self, 
        file: Any, 
        file_name: str,
        callback: Optional[callable] = None
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load large files in chunks with progress callback.
        
        Args:
            file: File-like object
            file_name: Original filename
            callback: Progress callback function(current_chunk, total_estimated)
            
        Returns:
            Tuple of (DataFrame, error_message)
        """
        try:
            file_extension = self._get_extension(file_name)
            
            if file_extension != '.csv':
                # Excel doesn't support chunking well, use regular load
                return self.load_file(file, file_name)
            
            chunks = []
            chunk_count = 0
            
            for chunk in pd.read_csv(file, chunksize=self.chunk_size):
                chunks.append(chunk)
                chunk_count += 1
                if callback:
                    callback(chunk_count, None)
            
            df = pd.concat(chunks, ignore_index=True)
            return df, None
            
        except Exception as e:
            logger.error(f"Error loading file in chunks: {str(e)}")
            return None, f"Error loading file: {str(e)}"
    
    def export_to_csv(self, df: pd.DataFrame) -> bytes:
        """
        Export DataFrame to CSV bytes.
        
        Args:
            df: DataFrame to export
            
        Returns:
            CSV content as bytes
        """
        return df.to_csv(index=False).encode('utf-8')
    
    def export_to_excel(self, df: pd.DataFrame) -> bytes:
        """
        Export DataFrame to Excel bytes.
        
        Args:
            df: DataFrame to export
            
        Returns:
            Excel content as bytes
        """
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Reconciled')
        return buffer.getvalue()
