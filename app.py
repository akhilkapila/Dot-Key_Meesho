"""
Streamlit Reconciliation App
A production-ready web-based reconciliation tool for matching and merging
sales and settlement data with customizable matching logic.

Author: Generated for Dot&Key Meesho Reconciliation
"""
import os
import sys
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import uuid

import streamlit as st
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.file_handler import FileHandler, FileStats
from utils.profile_manager import ProfileManager, MatchRule, PopulateRule, ProfileData
from utils.matching_engine import MatchingEngine, MatchStats
from utils.helpers import format_bytes, format_number, get_timestamp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Reconciliation App",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .stTitle {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3548 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Upload box styling */
    .upload-box {
        background: linear-gradient(135deg, #1e2738 0%, #2a3449 100%);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-box:hover {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Success/Error indicators */
    .status-matched {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-sales-only {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .status-settlement-only {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d3548 100%);
        border-radius: 8px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f2e;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Rule builder styling */
    .rule-item {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # File data
        'sales_df': None,
        'settlement_df': None,
        'credit_note_df': None,
        'reconciled_sales_df': None,
        'reconciled_settlement_df': None,
        'reconciled_credit_note_df': None,
        'workbook_created': False,
        
        # File info
        'sales_file_name': None,
        'settlement_file_name': None,
        'credit_note_file_name': None,
        'sales_stats': None,
        'settlement_stats': None,
        'credit_note_stats': None,
        
        # Matching configuration
        'match_rules': [],
        'populate_rules': [],
        
        # Current profile
        'current_profile_id': None,
        'current_profile_name': None,
        
        # Match stats
        'match_stats': None,
        
        # UI state
        'active_tab': 0,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
@st.cache_resource
def get_file_handler() -> FileHandler:
    """Get cached FileHandler instance."""
    return FileHandler()


@st.cache_resource
def get_profile_manager() -> ProfileManager:
    """Get cached ProfileManager instance."""
    os.makedirs('data', exist_ok=True)
    return ProfileManager('data/profiles.db')


@st.cache_resource
def get_matching_engine() -> MatchingEngine:
    """Get cached MatchingEngine instance."""
    return MatchingEngine()


def generate_rule_id() -> str:
    """Generate unique rule ID."""
    return str(uuid.uuid4())[:8]


# =============================================================================
# SIDEBAR - PROFILE MANAGEMENT
# =============================================================================
def render_sidebar():
    """Render the sidebar with profile management."""
    profile_manager = get_profile_manager()
    
    st.sidebar.markdown("## 🔧 Profile Management")
    st.sidebar.markdown("---")
    
    # Get all profiles
    profiles = profile_manager.get_all_profiles()
    profile_names = ["-- Select Profile --"] + [p['name'] for p in profiles]
    
    # Profile selector
    selected_profile = st.sidebar.selectbox(
        "📁 Load Profile",
        profile_names,
        key="profile_selector"
    )
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("📥 Load", use_container_width=True, disabled=selected_profile == "-- Select Profile --"):
            load_profile(selected_profile, profile_manager)
    
    with col2:
        if st.button("🗑️ Delete", use_container_width=True, disabled=selected_profile == "-- Select Profile --"):
            delete_profile(selected_profile, profile_manager)
    
    st.sidebar.markdown("---")
    
    # Save new profile
    st.sidebar.markdown("### 💾 Save Profile")
    new_profile_name = st.sidebar.text_input(
        "Profile Name",
        placeholder="e.g., Amazon Pay Recon",
        key="new_profile_name"
    )
    
    if st.sidebar.button("💾 Save Current Profile", use_container_width=True):
        save_current_profile(new_profile_name, profile_manager)
    
    st.sidebar.markdown("---")
    
    # Current profile info
    if st.session_state.current_profile_name:
        st.sidebar.info(f"📌 Active: **{st.session_state.current_profile_name}**")
    
    # Sample data loader
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧪 Test Data")
    if st.sidebar.button("📊 Load Sample Data", use_container_width=True):
        load_sample_data()


def load_profile(profile_name: str, profile_manager: ProfileManager):
    """Load a profile by name."""
    profile = profile_manager.get_profile_by_name(profile_name)
    if profile:
        st.session_state.match_rules = [r.to_dict() for r in profile.match_rules]
        st.session_state.populate_rules = [r.to_dict() for r in profile.populate_rules]
        st.session_state.selected_output_columns = profile.output_columns
        st.session_state.current_profile_id = profile.id
        st.session_state.current_profile_name = profile.name
        profile_manager.set_last_used(profile.id)
        st.success(f"✅ Loaded profile: {profile_name}")
        st.rerun()
    else:
        st.error("Profile not found")


def delete_profile(profile_name: str, profile_manager: ProfileManager):
    """Delete a profile by name."""
    profile = profile_manager.get_profile_by_name(profile_name)
    if profile:
        if profile_manager.delete_profile(profile.id):
            if st.session_state.current_profile_id == profile.id:
                st.session_state.current_profile_id = None
                st.session_state.current_profile_name = None
            st.success(f"🗑️ Deleted profile: {profile_name}")
            st.rerun()
        else:
            st.error("Failed to delete profile")


def save_current_profile(name: str, profile_manager: ProfileManager):
    """Save current configuration as a profile."""
    if not name or not name.strip():
        st.sidebar.error("Please enter a profile name")
        return
    
    # Get current headers
    sales_headers = list(st.session_state.sales_df.columns) if st.session_state.sales_df is not None else []
    settlement_headers = list(st.session_state.settlement_df.columns) if st.session_state.settlement_df is not None else []
    
    # Convert rules to dataclass objects
    match_rules = [MatchRule.from_dict(r) for r in st.session_state.match_rules]
    populate_rules = [PopulateRule.from_dict(r) for r in st.session_state.populate_rules]
    
    # Check if profile exists
    if profile_manager.profile_exists(name):
        profile = profile_manager.get_profile_by_name(name)
        if profile:
            profile_manager.update_profile(
                profile.id,
                sales_headers=sales_headers,
                settlement_headers=settlement_headers,
                match_rules=match_rules,
                populate_rules=populate_rules,
                output_columns=st.session_state.selected_output_columns
            )
            st.session_state.current_profile_id = profile.id
            st.session_state.current_profile_name = name
            st.sidebar.success(f"✅ Updated profile: {name}")
    else:
        profile_id = profile_manager.create_profile(
            name=name,
            sales_headers=sales_headers,
            settlement_headers=settlement_headers,
            match_rules=match_rules,
            populate_rules=populate_rules,
            output_columns=[]  # No longer used in new workflow
        )
        if profile_id:
            st.session_state.current_profile_id = profile_id
            st.session_state.current_profile_name = name
            profile_manager.set_last_used(profile_id)
            st.sidebar.success(f"✅ Saved profile: {name}")
        else:
            st.sidebar.error("Failed to save profile")
    
    st.rerun()


def load_sample_data():
    """Load sample data for testing."""
    file_handler = get_file_handler()
    
    try:
        # Load sample files
        sales_path = 'data/sample_sales.csv'
        settlement_path = 'data/sample_settlement.csv'
        
        if os.path.exists(sales_path) and os.path.exists(settlement_path):
            sales_df = pd.read_csv(sales_path)
            settlement_df = pd.read_csv(settlement_path)
            
            st.session_state.sales_df = sales_df
            st.session_state.settlement_df = settlement_df
            st.session_state.sales_file_name = 'sample_sales.csv'
            st.session_state.settlement_file_name = 'sample_settlement.csv'
            
            # Set stats for both files
            sales_size = os.path.getsize(sales_path)
            settlement_size = os.path.getsize(settlement_path)
            st.session_state.sales_stats = file_handler.get_file_stats(sales_df, sales_size)
            st.session_state.settlement_stats = file_handler.get_file_stats(settlement_df, settlement_size)
            
            st.success("✅ Sample data loaded!")
            st.rerun()
        else:
            st.error("Sample data files not found")
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")


# =============================================================================
# TAB 1: FILE UPLOAD
# =============================================================================
def render_upload_tab():
    """Render the file upload interface."""
    st.markdown("## 📤 Upload Files")
    st.markdown("Upload your sales, settlement, and optionally credit note files to begin reconciliation.")
    
    file_handler = get_file_handler()
    
    col1, col2, col3 = st.columns(3)
    
    # Sales File Upload
    with col1:
        st.markdown("### 📊 Sales File")
        render_file_uploader(
            "sales",
            file_handler,
            "sales_df",
            "sales_file_name",
            "sales_stats",
            "sales_key"
        )
    
    # Settlement File Upload
    with col2:
        st.markdown("### 💰 Settlement File")
        render_file_uploader(
            "settlement",
            file_handler,
            "settlement_df",
            "settlement_file_name",
            "settlement_stats",
            "settlement_key"
        )
    
    # Credit Note File Upload (Optional)
    with col3:
        st.markdown("### 📝 Credit Note File")
        st.caption("*(Optional)*")
        render_file_uploader(
            "credit_note",
            file_handler,
            "credit_note_df",
            "credit_note_file_name",
            "credit_note_stats",
            "credit_note_key"
        )
    
    st.markdown("---")
    
    # Merge Section
    if st.session_state.sales_df is not None and st.session_state.settlement_df is not None:
        render_merge_section()


def render_file_uploader(
    file_type: str,
    file_handler: FileHandler,
    df_key: str,
    name_key: str,
    stats_key: str,
    key_col_key: str
):
    """Render a file uploader component."""
    # Check if we already have data loaded (from sample or previous upload)
    has_existing_data = st.session_state[df_key] is not None
    
    uploaded_file = st.file_uploader(
        f"Choose {file_type} file",
        type=['csv', 'xlsx', 'xls'],
        key=f"{file_type}_uploader",
        help=f"Upload your {file_type} data file (CSV or Excel)"
    )
    
    # Only process new upload if file is uploaded AND it's a new file
    if uploaded_file is not None:
        current_file_name = st.session_state.get(name_key)
        # Check if this is a new file or the same file
        if current_file_name != uploaded_file.name:
            # New file uploaded - process it
            with st.spinner(f"Loading {file_type} file..."):
                df, error = file_handler.load_file(uploaded_file, uploaded_file.name)
            
            if error:
                st.error(f"❌ {error}")
                return
            
            # Validate
            is_valid, validation_error = file_handler.validate_file(df)
            if not is_valid:
                st.error(f"❌ {validation_error}")
                return
            
            # Store in session state
            st.session_state[df_key] = df
            st.session_state[name_key] = uploaded_file.name
            
            # Get stats
            stats = file_handler.get_file_stats(df, uploaded_file.size)
            st.session_state[stats_key] = stats
            
            # Auto-detect key columns and set default
            potential_keys = file_handler.detect_key_columns(df)
            if potential_keys:
                st.session_state[key_col_key] = potential_keys[0]
            elif len(df.columns) > 0:
                st.session_state[key_col_key] = df.columns[0]
    
    # Display data if we have it (either from upload or sample data)
    if st.session_state[df_key] is not None:
        df = st.session_state[df_key]
        file_name = st.session_state[name_key]
        stats = st.session_state[stats_key]
        
        st.success(f"✅ Loaded: {file_name}")
        
        # Display stats
        if stats:
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Rows", format_number(stats.rows))
            with metrics_col2:
                st.metric("Columns", stats.columns)
            with metrics_col3:
                st.metric("Size", format_bytes(stats.size_bytes))
        
        # Key column selector
        st.markdown("**Select Key Column for Matching:**")
        columns_list = list(df.columns)
        current_key = st.session_state.get(key_col_key)
        
        # Determine default index
        if current_key and current_key in columns_list:
            default_idx = columns_list.index(current_key)
        else:
            # Auto-detect potential keys
            potential_keys = file_handler.detect_key_columns(df)
            if potential_keys and potential_keys[0] in columns_list:
                default_idx = columns_list.index(potential_keys[0])
            else:
                default_idx = 0
        
        selected_key = st.selectbox(
            "Key Column",
            columns_list,
            index=default_idx,
            key=f"{file_type}_key_selector",
            help="Select the column to use for matching records"
        )
        st.session_state[key_col_key] = selected_key
        
        # Show potential keys hint
        potential_keys = file_handler.detect_key_columns(df)
        if potential_keys:
            st.caption(f"💡 Suggested keys: {', '.join(potential_keys[:3])}")
        
        # Preview
        with st.expander("📋 Preview (First 5 Rows)", expanded=False):
            st.dataframe(
                file_handler.get_preview(df, 5),
                use_container_width=True,
                hide_index=True
            )


def render_merge_section():
    """Render the optional raw data download section."""
    st.markdown("---")
    
    with st.expander("📁 Download Raw Data (Optional)", expanded=False):
        st.markdown("""
        Download both files combined into a single Excel workbook **before** processing.
        This is useful for backup or reviewing the raw data.
        """)
        
        # Only generate workbook when button is clicked (using session state)
        if 'raw_workbook_data' not in st.session_state:
            if st.button("📦 Generate Raw Data File", key="gen_raw_data"):
                with st.spinner("Generating workbook..."):
                    matching_engine = get_matching_engine()
                    st.session_state.raw_workbook_data = matching_engine.create_workbook(
                        st.session_state.sales_df,
                        st.session_state.settlement_df
                    )
                st.rerun()
        else:
            st.download_button(
                "📥 Download Raw Data (Excel)",
                data=st.session_state.raw_workbook_data,
                file_name="raw_data_combined.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


# =============================================================================
# TAB 2: LOGIC BUILDER
# =============================================================================
def render_logic_builder_tab():
    """Render the matching logic builder interface."""
    st.markdown("## ⚙️ Matching Logic Builder")
    
    if st.session_state.sales_df is None or st.session_state.settlement_df is None:
        st.warning("⚠️ Please upload both files first in the **Upload** tab.")
        return
    
    # Get column lists
    sales_columns = list(st.session_state.sales_df.columns)
    settlement_columns = list(st.session_state.settlement_df.columns)
    
    # Match Rules Section
    st.markdown("### 🎯 Match Rules")
    st.markdown("Define how to match records between sales and settlement data.")
    
    render_match_rules(sales_columns, settlement_columns)
    
    st.markdown("---")
    
    # Population Rules Section
    st.markdown("### 📝 Population Rules")
    st.markdown("Define how to populate columns when records match.")
    
    render_population_rules(sales_columns, settlement_columns)
    
    st.markdown("---")
    
    # Test Match Section
    render_test_match()


def render_match_rules(sales_columns: List[str], settlement_columns: List[str]):
    """Render match rules builder."""
    
    # Add new rule button
    if st.button("➕ Add Match Rule", key="add_match_rule"):
        new_rule = {
            'id': generate_rule_id(),
            'sales_column': sales_columns[0] if sales_columns else '',
            'settlement_column': settlement_columns[0] if settlement_columns else '',
            'match_type': 'exact',
            'tolerance': 0.01,
            'fuzzy_threshold': 80
        }
        st.session_state.match_rules.append(new_rule)
        st.rerun()
    
    # Display existing rules
    rules_to_remove = []
    
    for idx, rule in enumerate(st.session_state.match_rules):
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
            
            with col1:
                sales_col = st.selectbox(
                    "Sales Column",
                    sales_columns,
                    index=sales_columns.index(rule['sales_column']) if rule['sales_column'] in sales_columns else 0,
                    key=f"match_sales_{rule['id']}"
                )
                rule['sales_column'] = sales_col
            
            with col2:
                settlement_col = st.selectbox(
                    "Settlement Column",
                    settlement_columns,
                    index=settlement_columns.index(rule['settlement_column']) if rule['settlement_column'] in settlement_columns else 0,
                    key=f"match_settlement_{rule['id']}"
                )
                rule['settlement_column'] = settlement_col
            
            with col3:
                match_type = st.selectbox(
                    "Match Type",
                    ['exact', 'fuzzy', 'numeric_range', 'date_range'],
                    index=['exact', 'fuzzy', 'numeric_range', 'date_range'].index(rule['match_type']),
                    key=f"match_type_{rule['id']}"
                )
                rule['match_type'] = match_type
            
            with col4:
                if match_type == 'fuzzy':
                    threshold = st.slider(
                        "Threshold",
                        0, 100, rule.get('fuzzy_threshold', 80),
                        key=f"fuzzy_threshold_{rule['id']}"
                    )
                    rule['fuzzy_threshold'] = threshold
                elif match_type in ['numeric_range', 'date_range']:
                    tolerance = st.number_input(
                        "Tolerance" + (" (days)" if match_type == 'date_range' else " (%)"),
                        value=float(rule.get('tolerance', 0.01 if match_type == 'numeric_range' else 1)),
                        step=0.01 if match_type == 'numeric_range' else 1.0,
                        key=f"tolerance_{rule['id']}"
                    )
                    rule['tolerance'] = tolerance
                else:
                    st.markdown("*Exact match*")
            
            with col5:
                if st.button("🗑️", key=f"remove_match_{rule['id']}"):
                    rules_to_remove.append(idx)
    
    # Remove marked rules
    for idx in reversed(rules_to_remove):
        st.session_state.match_rules.pop(idx)
        st.rerun()
    
    if not st.session_state.match_rules:
        st.info("💡 Add match rules to define how records should be matched.")


def render_population_rules(sales_columns: List[str], settlement_columns: List[str]):
    """Render population rules builder with sheet and column letter selection."""
    from utils.matching_engine import get_column_letters_with_names, index_to_col_letter
    
    # Get Credit Note columns if available
    credit_note_columns = list(st.session_state.credit_note_df.columns) if st.session_state.credit_note_df is not None else []
    
    # Build available sheets list
    available_sheets = ['Sales', 'Settlement']
    if st.session_state.credit_note_df is not None:
        available_sheets.append('Credit Note')
    
    # Helper to get columns for a sheet
    def get_columns_for_sheet(sheet_name):
        if sheet_name == 'Sales':
            return sales_columns
        elif sheet_name == 'Settlement':
            return settlement_columns
        elif sheet_name == 'Credit Note':
            return credit_note_columns
        return []
    
    # Generate column letter options for each sheet
    sales_col_options = get_column_letters_with_names(st.session_state.sales_df)
    settlement_col_options = get_column_letters_with_names(st.session_state.settlement_df)
    
    # Also add new column options (next available letters)
    next_sales_letter = index_to_col_letter(len(sales_columns))
    next_settlement_letter = index_to_col_letter(len(settlement_columns))
    
    st.markdown("When a match is found, populate the target column with the value from the source column.")
    
    # Add new rule button
    if st.button("➕ Add Population Rule", key="add_populate_rule"):
        new_rule = {
            'id': generate_rule_id(),
            'target_sheet': 'Sales',
            'target_column_letter': next_sales_letter,
            'target_column_name': '',  # Custom column name
            'source_sheet': 'Settlement',
            'source_column': settlement_columns[0] if settlement_columns else ''
        }
        st.session_state.populate_rules.append(new_rule)
        st.rerun()
    
    # Display existing rules
    rules_to_remove = []
    
    for idx, rule in enumerate(st.session_state.populate_rules):
        with st.container():
            st.markdown(f"**Rule {idx + 1}**")
            col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1, 1.3, 1.2, 1.8, 0.5])
            
            with col1:
                target_sheet = st.selectbox(
                    "Target Sheet",
                    available_sheets,
                    index=available_sheets.index(rule.get('target_sheet', 'Sales')) if rule.get('target_sheet', 'Sales') in available_sheets else 0,
                    key=f"pop_target_sheet_{rule['id']}"
                )
                rule['target_sheet'] = target_sheet
            
            with col2:
                # Allow user to type any column letter
                current_letter = rule.get('target_column_letter', 'A')
                
                target_col = st.text_input(
                    "Target Col",
                    value=current_letter,
                    key=f"pop_target_col_{rule['id']}",
                    max_chars=3
                )
                rule['target_column_letter'] = target_col.upper().strip() if target_col else 'A'
            
            with col3:
                # Custom column name (optional)
                current_name = rule.get('target_column_name', '')
                col_name = st.text_input(
                    "Column Name",
                    value=current_name,
                    key=f"pop_col_name_{rule['id']}",
                    placeholder="Optional"
                )
                rule['target_column_name'] = col_name
            
            with col4:
                # Manual source sheet selection
                current_source_sheet = rule.get('source_sheet', 'Settlement')
                if current_source_sheet not in available_sheets:
                    current_source_sheet = available_sheets[0]
                
                source_sheet = st.selectbox(
                    "Source Sheet",
                    available_sheets,
                    index=available_sheets.index(current_source_sheet),
                    key=f"pop_source_sheet_{rule['id']}"
                )
                rule['source_sheet'] = source_sheet
            
            with col5:
                # Source column from source sheet
                source_cols = get_columns_for_sheet(source_sheet)
                current_source = rule.get('source_column', '')
                
                source_idx = 0
                if current_source in source_cols:
                    source_idx = source_cols.index(current_source)
                
                source_col = st.selectbox(
                    "Source Column",
                    source_cols if source_cols else [''],
                    index=source_idx,
                    key=f"pop_source_col_{rule['id']}"
                )
                rule['source_column'] = source_col
            
            with col6:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"remove_pop_{rule['id']}"):
                    rules_to_remove.append(idx)
            
            st.markdown("---")
    
    # Remove marked rules
    for idx in reversed(rules_to_remove):
        st.session_state.populate_rules.pop(idx)
        st.rerun()
    
    if not st.session_state.populate_rules:
        st.info("💡 Add population rules to define how columns should be filled when matches are found.")


def render_test_match():
    """Render test match section."""
    st.markdown("### 🧪 Test Match")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        sample_size = st.number_input(
            "Sample Size",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            key="test_sample_size"
        )
    
    with col2:
        if st.button("▶️ Run Test Match", type="primary"):
            run_test_match(sample_size)
    
    # Show test results
    if st.session_state.match_stats:
        stats = st.session_state.match_stats
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Match Rate",
                f"{stats.match_percentage:.1f}%",
                delta="Success" if stats.match_percentage > 50 else "Low"
            )
        with col2:
            st.metric("Matched", stats.matched_rows)
        with col3:
            st.metric("Unmatched", stats.unmatched_rows)
        with col4:
            st.metric("Multiple Matches", stats.multiple_match_rows)


def run_test_match(sample_size: int):
    """Run test match on sample data."""
    matching_engine = get_matching_engine()
    
    if not st.session_state.match_rules:
        st.warning("⚠️ Please add at least one match rule.")
        return
    
    with st.spinner("Testing match logic..."):
        try:
            stats = matching_engine.test_match(
                st.session_state.sales_df,
                st.session_state.settlement_df,
                st.session_state.match_rules,
                sample_size
            )
            
            st.session_state.match_stats = stats
            st.success(f"✅ Test complete: {stats.match_percentage:.1f}% match rate")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Test failed: {str(e)}")
            logger.error(f"Test match error: {str(e)}")


# =============================================================================
# TAB 3: PREVIEW
# =============================================================================
def render_preview_tab():
    """Render the preview tab showing both sheets."""
    st.markdown("## 👁️ Preview Data")
    
    if st.session_state.sales_df is None or st.session_state.settlement_df is None:
        st.warning("⚠️ Please upload both files first in the **Upload** tab.")
        return
    
    # Show both sheets in tabs
    sheet_tab1, sheet_tab2 = st.tabs(["📊 Sales Sheet", "💰 Settlement Sheet"])
    
    with sheet_tab1:
        st.markdown(f"### Sales Data ({len(st.session_state.sales_df)} rows)")
        st.dataframe(
            st.session_state.sales_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    
    with sheet_tab2:
        st.markdown(f"### Settlement Data ({len(st.session_state.settlement_df)} rows)")
        st.dataframe(
            st.session_state.settlement_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    
    # Show reconciled data if available
    if st.session_state.reconciled_sales_df is not None:
        st.markdown("---")
        st.markdown("## ✅ Reconciled Data")
        
        recon_tab1, recon_tab2 = st.tabs(["📊 Reconciled Sales", "💰 Reconciled Settlement"])
        
        with recon_tab1:
            st.dataframe(
                st.session_state.reconciled_sales_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        
        with recon_tab2:
            st.dataframe(
                st.session_state.reconciled_settlement_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )


# =============================================================================
# TAB 4: EXPORT (Run Reconciliation & Download)
# =============================================================================
def render_export_tab():
    """Render the export tab with reconciliation execution."""
    st.markdown("## 📥 Run Reconciliation & Export")
    
    if st.session_state.sales_df is None or st.session_state.settlement_df is None:
        st.warning("⚠️ Please upload both files first in the **Upload** tab.")
        return
    
    # Summary of configuration
    st.markdown("### 📋 Configuration Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Match Rules", len(st.session_state.match_rules))
    with col2:
        st.metric("Population Rules", len(st.session_state.populate_rules))
    
    # Show rules summary
    if st.session_state.match_rules:
        with st.expander("🎯 Match Rules", expanded=False):
            for idx, rule in enumerate(st.session_state.match_rules):
                st.write(f"**Rule {idx + 1}**: Sales.`{rule['sales_column']}` ↔ Settlement.`{rule['settlement_column']}` ({rule['match_type']})")
    
    if st.session_state.populate_rules:
        with st.expander("📝 Population Rules", expanded=False):
            for idx, rule in enumerate(st.session_state.populate_rules):
                st.write(f"**Rule {idx + 1}**: {rule.get('target_sheet', 'Sales')}[{rule.get('target_column_letter', 'A')}] ← {rule.get('source_sheet', 'Settlement')}.`{rule.get('source_column', '')}`")
    
    st.markdown("---")
    
    # Run Reconciliation Button
    st.markdown("### ▶️ Execute Reconciliation")
    
    if st.button("🚀 Run Reconciliation", type="primary", use_container_width=True):
        run_reconciliation()
    
    st.markdown("---")
    
    # Export Section
    st.markdown("### 📥 Download Results")
    
    if st.session_state.reconciled_sales_df is not None:
        # Show success message
        st.success(f"✅ **Reconciliation Complete!** Ready to download.")
        
        matching_engine = get_matching_engine()
        
        # Show stats
        if st.session_state.match_stats:
            stats = st.session_state.match_stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Match Rate", f"{stats.match_percentage:.1f}%")
            with col2:
                st.metric("Matched Rows", f"{stats.matched_rows:,}")
            with col3:
                st.metric("Unmatched", f"{stats.unmatched_rows:,}")
            with col4:
                st.metric("Multiple Matches", f"{stats.multiple_match_rows:,}")
        
        st.markdown("---")
        
        # Filename options
        timestamp_filename = st.checkbox("Add Timestamp to Filename", value=True, key="timestamp_filename")
        
        base_name = "reconciled_workbook"
        if timestamp_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"reconciled_workbook_{timestamp}"
        
        # Download button with error handling
        try:
            with st.spinner("Preparing download file..."):
                workbook_data = matching_engine.export_reconciled_workbook(
                    st.session_state.reconciled_sales_df,
                    st.session_state.reconciled_settlement_df
                )
            
            st.download_button(
                "📥 Download Reconciled Workbook (Excel)",
                data=workbook_data,
                file_name=f"{base_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )
        except Exception as e:
            st.error(f"❌ Error generating download: {str(e)}")
    else:
        st.info("💡 Click 'Run Reconciliation' above to process the data and enable download.")


def run_reconciliation():
    """Execute the reconciliation process."""
    matching_engine = get_matching_engine()
    
    if not st.session_state.match_rules:
        st.warning("⚠️ Please add at least one match rule in the Logic Builder tab.")
        return
    
    # Show processing info
    total_sales = len(st.session_state.sales_df)
    total_settlement = len(st.session_state.settlement_df)
    
    status_container = st.container()
    with status_container:
        st.info(f"🔄 Processing {total_sales:,} sales rows against {total_settlement:,} settlement rows...")
        st.markdown(f"**Match Rules:** {len(st.session_state.match_rules)} | **Population Rules:** {len(st.session_state.populate_rules)}")
    
    with st.spinner(f"Running reconciliation on {total_sales:,} × {total_settlement:,} records..."):
        try:
            # Run reconciliation
            reconciled_sales, reconciled_settlement, stats = matching_engine.run_reconciliation(
                st.session_state.sales_df,
                st.session_state.settlement_df,
                st.session_state.match_rules,
                st.session_state.populate_rules
            )
            
            # Store results
            st.session_state.reconciled_sales_df = reconciled_sales
            st.session_state.reconciled_settlement_df = reconciled_settlement
            st.session_state.match_stats = stats
            
            st.success(f"""
            ✅ **Reconciliation Complete!**
            - **Total Rows Processed:** {stats.total_target_rows:,}
            - **Matched:** {stats.matched_rows:,} ({stats.match_percentage:.1f}%)
            - **Unmatched:** {stats.unmatched_rows:,}
            - **Multiple Matches:** {stats.multiple_match_rows:,}
            """)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Reconciliation failed: {str(e)}")
            logger.error(f"Reconciliation error: {str(e)}")


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("""
    <h1 style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5rem; margin-bottom: 0;">
    🔄 Reconciliation App
    </h1>
    <p style="text-align: center; color: #888; margin-top: 0.5rem;">
    Match and reconcile sales with settlement data
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Render sidebar
    render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload",
        "⚙️ Logic Builder",
        "👁️ Preview",
        "📥 Export"
    ])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_logic_builder_tab()
    
    with tab3:
        render_preview_tab()
    
    with tab4:
        render_export_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #666; font-size: 0.8rem;">
    Reconciliation App v1.0 | Built with Streamlit
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
