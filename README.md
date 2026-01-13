# Streamlit Reconciliation App

A production-ready web-based reconciliation tool for matching and merging sales and settlement data with customizable matching logic.

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run app.py
   ```

3. **Open browser:**
   Navigate to `http://localhost:8501`

### Streamlit Cloud Deployment

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Click "Deploy"

## 📁 Project Structure

```
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── utils/
│   ├── __init__.py
│   ├── file_handler.py       # File upload and parsing
│   ├── profile_manager.py    # SQLite profile CRUD
│   ├── matching_engine.py    # Core matching logic
│   └── helpers.py            # Utility functions
├── data/
│   ├── profiles.db           # SQLite database (auto-created)
│   ├── sample_sales.csv      # Sample test data
│   └── sample_settlement.csv # Sample test data
└── README.md
```

## ✨ Features

### 📤 Dual File Upload
- Upload CSV or Excel files
- Auto-validation and preview
- File statistics display
- Auto-detect potential key columns

### 🔗 Data Merge
- Full outer join on selected keys
- Status column (Matched, Sales Only, Settlement Only)
- Download merged raw data

### ⚙️ Matching Logic Builder
- **Exact Match**: Precise text matching
- **Fuzzy Match**: Configurable similarity threshold
- **Numeric Range**: Tolerance-based matching
- **Date Range**: Day-based tolerance
- **Population Rules**: Populate columns based on conditions

### 💾 Profile Management
- Save/Load matching configurations
- SQLite-based persistence
- Auto-save last used profile

### 📥 Export Options
- CSV or Excel format
- Column selection
- Filter by match status
- Timestamped filenames

## 🧪 Testing

Load sample data using the sidebar button to test all features without uploading files.

## 🔧 Configuration

Edit `.streamlit/config.toml` for:
- Theme customization
- Upload size limits
- Server settings

## 📋 Requirements

- Python 3.8+
- Streamlit 1.28+
- Pandas 2.0+
- SQLAlchemy 2.0+
- PyArrow 14.0+

## 📄 License

MIT License
