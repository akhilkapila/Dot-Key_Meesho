"""
Matching Engine Module for the Reconciliation App.
Core logic for cross-sheet matching and population.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import io

import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from openpyxl import Workbook
from openpyxl.utils import get_column_letter, column_index_from_string

logger = logging.getLogger(__name__)


def col_letter_to_index(letter: str) -> int:
    """Convert Excel column letter to 0-based index."""
    return column_index_from_string(letter.upper()) - 1


def index_to_col_letter(index: int) -> str:
    """Convert 0-based index to Excel column letter."""
    return get_column_letter(index + 1)


def get_column_letters_with_names(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Get list of (column_letter, column_name) tuples for a DataFrame.
    
    Returns:
        List of tuples like [('A', 'order_id'), ('B', 'amount'), ...]
    """
    result = []
    for idx, col_name in enumerate(df.columns):
        letter = index_to_col_letter(idx)
        result.append((letter, str(col_name)))
    return result


@dataclass
class MatchStats:
    """Statistics for matching results."""
    total_target_rows: int
    matched_rows: int
    unmatched_rows: int
    multiple_match_rows: int
    match_percentage: float
    
    def to_dict(self) -> Dict:
        return {
            'total_target_rows': self.total_target_rows,
            'matched_rows': self.matched_rows,
            'unmatched_rows': self.unmatched_rows,
            'multiple_match_rows': self.multiple_match_rows,
            'match_percentage': round(self.match_percentage, 2)
        }


class MatchingEngine:
    """
    Core matching engine for cross-sheet reconciliation operations.
    Supports matching between two sheets and populating values based on matches.
    """
    
    def __init__(self):
        """Initialize MatchingEngine."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_workbook(
        self,
        sales_df: pd.DataFrame,
        settlement_df: pd.DataFrame
    ) -> bytes:
        """
        Create a multi-sheet Excel workbook with Sales and Settlement data.
        
        Args:
            sales_df: Sales DataFrame
            settlement_df: Settlement DataFrame
            
        Returns:
            Excel file as bytes
        """
        self.logger.info("Creating multi-sheet workbook")
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            sales_df.to_excel(writer, sheet_name='Sales', index=False)
            settlement_df.to_excel(writer, sheet_name='Settlement', index=False)
        
        return buffer.getvalue()
    
    def find_matches(
        self,
        target_df: pd.DataFrame,
        target_col: str,
        source_df: pd.DataFrame,
        source_col: str,
        match_type: str = 'exact',
        fuzzy_threshold: int = 80,
        tolerance: float = 0.01
    ) -> Dict[int, List[int]]:
        """
        Find matching rows between two DataFrames.
        
        Args:
            target_df: Target DataFrame (where we want to populate)
            target_col: Column name in target to match on
            source_df: Source DataFrame (where we get values from)
            source_col: Column name in source to match on
            match_type: 'exact', 'fuzzy', 'numeric_range', 'date_range'
            fuzzy_threshold: Threshold for fuzzy matching (0-100)
            tolerance: Tolerance for numeric range matching
            
        Returns:
            Dict mapping target row index to list of matching source row indices
        """
        matches = {}
        
        target_values = target_df[target_col].astype(str).str.strip().str.lower()
        source_values = source_df[source_col].astype(str).str.strip().str.lower()
        
        total_rows = len(target_values)
        
        for target_idx, target_val in enumerate(target_values):
            if pd.isna(target_val) or target_val == 'nan' or target_val == '':
                continue
                
            matching_source_indices = []
            
            for source_idx, source_val in enumerate(source_values):
                if pd.isna(source_val) or source_val == 'nan' or source_val == '':
                    continue
                
                is_match = False
                
                if match_type == 'exact':
                    is_match = (target_val == source_val)
                    
                elif match_type == 'fuzzy':
                    score = fuzz.ratio(target_val, source_val)
                    is_match = (score >= fuzzy_threshold)
                    
                elif match_type == 'numeric_range':
                    try:
                        t_num = float(target_df[target_col].iloc[target_idx])
                        s_num = float(source_df[source_col].iloc[source_idx])
                        if t_num != 0:
                            diff_ratio = abs(t_num - s_num) / abs(t_num)
                            is_match = (diff_ratio <= tolerance)
                        else:
                            is_match = (s_num == 0)
                    except (ValueError, TypeError):
                        is_match = False
                        
                elif match_type == 'date_range':
                    try:
                        t_date = pd.to_datetime(target_df[target_col].iloc[target_idx])
                        s_date = pd.to_datetime(source_df[source_col].iloc[source_idx])
                        diff_days = abs((t_date - s_date).days)
                        is_match = (diff_days <= int(tolerance))
                    except (ValueError, TypeError):
                        is_match = False
                
                if is_match:
                    matching_source_indices.append(source_idx)
            
            if matching_source_indices:
                matches[target_idx] = matching_source_indices
        
        return matches, total_rows
    
    def populate_values(
        self,
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
        matches: Dict[int, List[int]],
        target_col_letter: str,
        source_col_name: str
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Populate values from source to target based on matches.
        
        Args:
            target_df: Target DataFrame to modify
            source_df: Source DataFrame to get values from
            matches: Dict mapping target row index to source row indices
            target_col_letter: Excel column letter for target (A, B, C, ...)
            source_col_name: Column name in source DataFrame
            
        Returns:
            Tuple of (modified target DataFrame, stats dict)
        """
        result_df = target_df.copy()
        target_col_idx = col_letter_to_index(target_col_letter)
        
        # Ensure we have enough columns
        while len(result_df.columns) <= target_col_idx:
            new_col_name = f"Column_{len(result_df.columns) + 1}"
            result_df[new_col_name] = np.nan
        
        target_col_name = result_df.columns[target_col_idx]
        
        stats = {
            'populated': 0,
            'summed': 0,
            'errors': 0
        }
        
        for target_idx, source_indices in matches.items():
            if len(source_indices) == 1:
                # Single match - direct copy
                value = source_df[source_col_name].iloc[source_indices[0]]
                result_df.iloc[target_idx, target_col_idx] = value
                stats['populated'] += 1
                
            elif len(source_indices) > 1:
                # Multiple matches - try to sum
                values = [source_df[source_col_name].iloc[idx] for idx in source_indices]
                
                try:
                    # Try to convert to numeric and sum
                    numeric_values = [float(v) for v in values if pd.notna(v)]
                    if numeric_values:
                        result_df.iloc[target_idx, target_col_idx] = sum(numeric_values)
                        stats['summed'] += 1
                    else:
                        result_df.iloc[target_idx, target_col_idx] = "Data not numbers"
                        stats['errors'] += 1
                except (ValueError, TypeError):
                    result_df.iloc[target_idx, target_col_idx] = "Data not numbers"
                    stats['errors'] += 1
        
        return result_df, stats
    
    def run_reconciliation(
        self,
        sales_df: pd.DataFrame,
        settlement_df: pd.DataFrame,
        match_rules: List[Dict],
        populate_rules: List[Dict],
        progress_callback: Optional[callable] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, MatchStats]:
        """
        Run full reconciliation with matching and population.
        
        Args:
            sales_df: Sales DataFrame
            settlement_df: Settlement DataFrame
            match_rules: List of match rule dicts
            populate_rules: List of populate rule dicts
            progress_callback: Optional callback(current_row, total_rows, phase) for progress updates
            
        Returns:
            Tuple of (modified sales_df, modified settlement_df, stats)
        """
        result_sales = sales_df.copy()
        result_settlement = settlement_df.copy()
        
        total_sales_rows = len(sales_df)
        total_settlement_rows = len(settlement_df)
        
        # Build match lookup based on rules
        all_matches_to_sales = {}  # target_idx -> source_indices
        all_matches_to_settlement = {}
        
        # Report starting
        if progress_callback:
            progress_callback(0, total_sales_rows, "Starting matching...")
        
        for rule_idx, rule in enumerate(match_rules):
            sales_col = rule.get('sales_column')
            settlement_col = rule.get('settlement_column')
            match_type = rule.get('match_type', 'exact')
            fuzzy_threshold = rule.get('fuzzy_threshold', 80)
            tolerance = rule.get('tolerance', 0.01)
            
            if progress_callback:
                progress_callback(0, total_sales_rows, f"Matching rule {rule_idx + 1}: {sales_col} ↔ {settlement_col}")
            
            # Find matches from Sales perspective
            matches, _ = self.find_matches(
                result_sales, sales_col,
                result_settlement, settlement_col,
                match_type, fuzzy_threshold, tolerance
            )
            
            # Merge matches
            for target_idx, source_indices in matches.items():
                if target_idx not in all_matches_to_sales:
                    all_matches_to_sales[target_idx] = []
                all_matches_to_sales[target_idx].extend(source_indices)
            
            if progress_callback:
                progress_callback(len(all_matches_to_sales), total_sales_rows, f"Found {len(all_matches_to_sales)} matches so far...")
            
            # Find matches from Settlement perspective
            reverse_matches, _ = self.find_matches(
                result_settlement, settlement_col,
                result_sales, sales_col,
                match_type, fuzzy_threshold, tolerance
            )
            
            for target_idx, source_indices in reverse_matches.items():
                if target_idx not in all_matches_to_settlement:
                    all_matches_to_settlement[target_idx] = []
                all_matches_to_settlement[target_idx].extend(source_indices)
        
        if progress_callback:
            progress_callback(len(all_matches_to_sales), total_sales_rows, "Matching complete. Applying population rules...")
        
        # Apply populate rules
        for rule in populate_rules:
            target_sheet = rule.get('target_sheet', 'Sales')
            target_col_letter = rule.get('target_column_letter', 'A')
            source_sheet = rule.get('source_sheet', 'Settlement')
            source_col = rule.get('source_column', '')
            
            if target_sheet == 'Sales':
                target_df = result_sales
                source_df = result_settlement
                matches = all_matches_to_sales
            else:
                target_df = result_settlement
                source_df = result_sales
                matches = all_matches_to_settlement
            
            if source_col and source_col in source_df.columns:
                modified_df, stats = self.populate_values(
                    target_df, source_df, matches,
                    target_col_letter, source_col
                )
                
                if target_sheet == 'Sales':
                    result_sales = modified_df
                else:
                    result_settlement = modified_df
        
        # Calculate stats
        total_target = len(sales_df)
        matched = len(all_matches_to_sales)
        multiple = sum(1 for indices in all_matches_to_sales.values() if len(indices) > 1)
        
        match_stats = MatchStats(
            total_target_rows=total_target,
            matched_rows=matched,
            unmatched_rows=total_target - matched,
            multiple_match_rows=multiple,
            match_percentage=(matched / total_target * 100) if total_target > 0 else 0
        )
        
        return result_sales, result_settlement, match_stats
    
    def export_reconciled_workbook(
        self,
        sales_df: pd.DataFrame,
        settlement_df: pd.DataFrame
    ) -> bytes:
        """
        Export reconciled data as multi-sheet Excel workbook.
        
        Args:
            sales_df: Modified Sales DataFrame
            settlement_df: Modified Settlement DataFrame
            
        Returns:
            Excel file as bytes
        """
        return self.create_workbook(sales_df, settlement_df)
    
    def test_match(
        self,
        sales_df: pd.DataFrame,
        settlement_df: pd.DataFrame,
        match_rules: List[Dict],
        sample_size: int = 100
    ) -> MatchStats:
        """
        Test matching logic on a sample of data.
        
        Args:
            sales_df: Full sales DataFrame
            settlement_df: Full settlement DataFrame
            match_rules: Match rules to test
            sample_size: Number of rows to sample
            
        Returns:
            MatchStats object
        """
        # Sample data
        sales_sample = sales_df.head(sample_size)
        settlement_sample = settlement_df.head(sample_size)
        
        all_matches = {}
        
        for rule in match_rules:
            sales_col = rule.get('sales_column')
            settlement_col = rule.get('settlement_column')
            match_type = rule.get('match_type', 'exact')
            fuzzy_threshold = rule.get('fuzzy_threshold', 80)
            tolerance = rule.get('tolerance', 0.01)
            
            matches, _ = self.find_matches(
                sales_sample, sales_col,
                settlement_sample, settlement_col,
                match_type, fuzzy_threshold, tolerance
            )
            
            for target_idx, source_indices in matches.items():
                if target_idx not in all_matches:
                    all_matches[target_idx] = []
                all_matches[target_idx].extend(source_indices)
        
        total = len(sales_sample)
        matched = len(all_matches)
        multiple = sum(1 for indices in all_matches.values() if len(indices) > 1)
        
        return MatchStats(
            total_target_rows=total,
            matched_rows=matched,
            unmatched_rows=total - matched,
            multiple_match_rows=multiple,
            match_percentage=(matched / total * 100) if total > 0 else 0
        )
