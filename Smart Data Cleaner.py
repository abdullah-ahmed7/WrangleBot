import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import google.generativeai as genai

# ---------- Gemini API Integration ----------
try:
    if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    else:
        st.warning("Gemini API key not found in st.secrets['gemini']['api_key']. AI suggestions will be unavailable.")
        gemini_model = None
except Exception as e:
    st.warning(f"Failed to configure Gemini API: {e}. AI suggestions will be unavailable.")
    gemini_model = None

def get_ai_suggestion(prompt: str) -> str:
    """Get AI suggestion with reasoning for data cleaning steps"""
    if gemini_model:
        try:
            response = gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"⚠ Error from Gemini API: {e}"
    return "Gemini AI is not configured or available."

def get_duplicate_suggestion(df, dataset_name):
    """Get AI suggestion for handling duplicates"""
    dup_count = df.duplicated().sum()
    total_rows = len(df)
    dup_percentage = (dup_count / total_rows) * 100 if total_rows > 0 else 0
    
    prompt = f"""
    Dataset: {dataset_name}
    Total rows: {total_rows}
    Duplicate rows: {dup_count} ({dup_percentage:.1f}%)
    
    Provide a brief recommendation (2-3 sentences) on whether to remove duplicates and why, considering:
    1. The percentage of duplicates
    2. Potential impact on data analysis
    3. Best practices for this scenario
    
    Be concise and practical.
    """
    return get_ai_suggestion(prompt)

def get_missing_value_suggestion(df, column_name, dataset_name):
    """Get AI suggestion for handling missing values in a specific column"""
    col_data = df[column_name]
    missing_count = col_data.isnull().sum()
    total_count = len(col_data)
    missing_percentage = (missing_count / total_count) * 100 if total_count > 0 else 0
    data_type = str(col_data.dtype)
    unique_values = col_data.nunique()
    
    # Get sample values (non-null)
    sample_values = col_data.dropna().head(5).tolist()
    
    prompt = f"""
    Dataset: {dataset_name}
    Column: {column_name}
    Data type: {data_type}
    Missing values: {missing_count} out of {total_count} ({missing_percentage:.1f}%)
    Unique values: {unique_values}
    Sample values: {sample_values}
    
    Recommend the best strategy for handling missing values in this column:
    - Drop Rows
    - Fill with Mean (numeric only)
    - Fill with Median (numeric only)
    - Fill with Mode
    - Interpolate (numeric only)
    
    Provide your recommendation with a brief reason (2-3 sentences) considering:
    1. The percentage of missing values
    2. Data type and distribution
    3. Impact on analysis
    
    Format: "Recommendation: [Strategy] - Reason: [Brief explanation]"
    """
    return get_ai_suggestion(prompt)

def get_dtype_conversion_suggestion(df, column_name, dataset_name):
    """Get AI suggestion for data type conversion"""
    col_data = df[column_name]
    current_dtype = str(col_data.dtype)
    sample_values = col_data.dropna().head(10).tolist()
    unique_values = col_data.nunique()
    
    prompt = f"""
    Dataset: {dataset_name}
    Column: {column_name}
    Current data type: {current_dtype}
    Unique values: {unique_values}
    Sample values: {sample_values}
    
    Analyze if this column needs data type conversion. Consider:
    1. Current type vs. actual data content
    2. Potential issues with current type
    3. Best type for analysis/modeling
    
    Recommend whether to convert and to which type (string, int, float, datetime, or keep current).
    
    Format: "Recommendation: [Keep current/Convert to X] - Reason: [Brief explanation in 2-3 sentences]"
    """
    return get_ai_suggestion(prompt)

def get_outlier_suggestion(df, numeric_columns, dataset_name):
    """Get AI suggestion for outlier handling"""
    if not numeric_columns:
        return "No numeric columns available for outlier analysis."
    
    outlier_info = []
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100 if len(df) > 0 else 0
        outlier_info.append(f"{col}: {outlier_count} outliers ({outlier_percentage:.1f}%)")
    
    prompt = f"""
    Dataset: {dataset_name}
    Numeric columns with outliers (using IQR method):
    {chr(10).join(outlier_info)}
    
    Provide recommendations for outlier handling:
    1. Which columns should have outliers removed?
    2. Are there columns where outliers might be legitimate extreme values?
    3. What's the potential impact of removing outliers?
    
    Give a practical recommendation in 3-4 sentences.
    """
    return get_ai_suggestion(prompt)

def get_scaling_suggestion(df, numeric_columns, dataset_name):
    """Get AI suggestion for scaling/normalization"""
    if not numeric_columns:
        return "No numeric columns available for scaling."
    
    scale_info = []
    for col in numeric_columns:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            mean_val = col_data.mean()
            std_val = col_data.std()
            min_val = col_data.min()
            max_val = col_data.max()
            range_val = max_val - min_val
            scale_info.append(f"{col}: mean={mean_val:.2f}, std={std_val:.2f}, range={range_val:.2f}")
    
    prompt = f"""
    Dataset: {dataset_name}
    Numeric columns statistics:
    {chr(10).join(scale_info)}
    
    Analyze if scaling is needed and recommend:
    1. Should these columns be scaled?
    2. Standardization (Z-score) vs. Min-Max normalization - which is better?
    3. Which columns specifically need scaling?
    
    Consider typical use cases like machine learning, statistical analysis, etc.
    
    Format: "Recommendation: [Scale/Don't scale] with [method] - Reason: [Brief explanation]"
    """
    return get_ai_suggestion(prompt)

# ---------- Page Setup ----------
st.set_page_config(
    page_title="Smart Data Cleaning Assistant",
    layout="wide",
    page_icon="🤖"
)

# ---------- Global Custom CSS ----------
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #3b0a4e;
    }

    [data-testid="stSidebar"] button {
        background-color: transparent !important;
        color: white !important;
        border: none !important;
        box-shadow: none !important;
        text-align: left;
        font-size: 17px;
        padding: 10px 16px;
    }

    [data-testid="stSidebar"] button:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 6px;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .stApp {
        background-color: #f3e8ff;
    }

    /* ▼ Add arrow to dropdowns */
    div[data-baseweb="select"] > div::after {
        content: "▼";
        color: #6a0dad;
        position: absolute;
        right: 14px;
        top: 50%;
        transform: translateY(-50%);
        pointer-events: none;
        font-size: 16px;
    }

    .ai-suggestion-box {
        background-color: #e8f4fd;
        border-left: 4px solid #2196F3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 6px;
        font-size: 14px;
        color: #333;
    }

    .ai-suggestion-title {
        font-weight: 600;
        color: #1976D2;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .styled-table {
        font-family: "Segoe UI", sans-serif;
        border-collapse: collapse;
        width: 100%;
        margin-top: 20px;
        color: #222222;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Session State Initialization ----------
if 'view' not in st.session_state:
    st.session_state['view'] = 'home'
if 'datasets' not in st.session_state:
    st.session_state['datasets'] = []
if 'current_df' not in st.session_state:
    st.session_state['current_df'] = None
if 'current_df_name' not in st.session_state:
    st.session_state['current_df_name'] = None
if 'cleaned_dfs' not in st.session_state:
    st.session_state['cleaned_dfs'] = {}

# Helper function for static messages
def show_static_message(msg_type, msg_text):
    color_map = {
        "success": "#d4edda",
        "error": "#f8d7da",
        "info": "#cce5ff",
        "warning": "#fff3cd"
    }
    text_color = {
        "success": "#155724",
        "error": "#721c24",
        "info": "#004085",
        "warning": "#856404"
    }
    return f"""
        <div style='
            background-color: {color_map[msg_type]};
            color: {text_color[msg_type]};
            padding: 8px 16px;
            border-radius: 6px;
            margin: 10px 0;
            font-size: 14px;
        '>{msg_text}</div>
    """

# ---------- Sidebar Navigation ----------
with st.sidebar:
    st.markdown('<h3 style="color:white;">Smart Data Cleaning Assistant</h3>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("Home", key="home", use_container_width=True):
        st.session_state['view'] = 'home'; st.rerun()
    if st.button("How It Works", key="how", use_container_width=True):
        st.session_state['view'] = 'how_it_works'; st.rerun()
    if st.button("Upload Data", key="upload", use_container_width=True):
        st.session_state['view'] = 'upload'; st.rerun()
    if st.button("Explore Data", key="explore", use_container_width=True):
        st.session_state['view'] = 'explore'; st.rerun()
    if st.button("Clean Data", key="clean_data", use_container_width=True):
        st.session_state['view'] = 'clean_data'; st.rerun()
    if st.button("Download", key="download", use_container_width=True):
        st.session_state['view'] = 'download'; st.rerun()
    if st.button("About / Credits", key="about", use_container_width=True):
        st.session_state['view'] = 'about'; st.rerun()

# ---------- HOME ----------
if st.session_state['view'] == 'home':
    st.markdown("""
        <style>
        .home-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 50px 5%;
            background-color: transparent;
            color: black;
            flex-wrap: wrap;
        }

        .home-text {
            max-width: 50%;
            min-width: 300px;
        }

        .home-title {
            font-size: 48px;
            font-weight: 800;
            margin-bottom: 10px;
            color: #6a0dad;
        }

        .home-subtitle {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            color: #4b6cb7;
        }

        .home-quote {
            font-size: 18px;
            color: #333333;
            margin-bottom: 30px;
            line-height: 1.6;
        }
        </style>

        <div class="home-container">
            <div class="home-text">
                <div class="home-title">Welcome to Smart Data Cleaner</div>
                <div class="home-subtitle">AI WrangleBot</div>
                <div class="home-quote">
                    "In the vast ocean of data, insight is the treasure — but only when the clutter is cleared, the noise is filtered, and the mess is cleaned.
                    Behind every powerful machine learning model, every stunning dashboard, and every confident decision, there is clean, reliable data.
                    Data cleaning isn't just a step — it's the foundation of every intelligent solution."
                </div>
            </div>
            <div>
                <img src="https://cdn-icons-png.flaticon.com/512/3589/3589018.png" width="300">
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Get Started"):
        st.session_state['view'] = 'upload'
        st.rerun()

# ---------- UPLOAD PAGE ----------
elif st.session_state['view'] == 'upload':
    st.markdown("""
        <style>
        .upload-section {
            background-color: #e2e3ff;
            padding: 60px;
            margin: 50px 8%;
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            font-family: "Segoe UI", sans-serif;
        }
        .upload-title {
            font-size: 36px;
            font-weight: 800;
            color: #6a0dad;
            margin-bottom: 20px;
        }
        .upload-subtext {
            font-size: 18px;
            color: #333333;
            margin-bottom: 30px;
        }
        .styled-table {
            font-family: "Segoe UI", sans-serif;
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
            color: #222222;
        }
        .styled-table th {
            background-color: #6a0dad;
            color: white;
            padding: 10px;
        }
        .styled-table td {
            padding: 10px;
            border: 1px solid #ddd;
            background-color: #f5f2ff;
            color: #222222;
        }
        .styled-table tr:nth-child(even) {
            background-color: #ece6ff;
        }
        .styled-table tr:hover {
            background-color: #e0dbff;
        }
        label[data-testid="stSelectboxLabel"] {
          color: #6a0dad !important;
          font-size: 18px !important;
          font-weight: 700 !important;
          margin-bottom: 10px;
        }
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #000000 !important;
            border-radius: 8px !important;
            border: 1px solid #ccc !important;
        }
        div[data-baseweb="select"] div[role="button"] {
            color: #6a0dad !important;
            font-weight: 600;
        }
        ul[role="listbox"] {
            background-color: #ffffff !important;
            color: #000 !important;
            font-size: 16px;
        }
        li[role="option"]:hover {
            background-color: #e2e3ff !important;
        }
        </style>

        <div class="upload-section">
            <div class="upload-title">Upload Your Data</div>
            <div class="upload-subtext">Select one or more CSV files to begin cleaning with AI WrangleBot.</div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload one or more CSV files",
        type=["csv"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)
                existing_names = [d['name'] for d in st.session_state['datasets']]
                if uploaded_file.name not in existing_names:
                    st.session_state['datasets'].append({
                        "name": uploaded_file.name,
                        "dataframe": df
                    })
                    st.session_state['cleaned_dfs'][uploaded_file.name] = df.copy()
                    st.markdown(f"""
                        <div style="
                            background-color: #d4edda;
                            color: #155724;
                            padding: 15px 20px;
                            border-left: 6px solid #28a745;
                            border-radius: 8px;
                            font-size: 16px;
                            margin-top: 10px;
                            margin-bottom: 15px;
                        ">
                            ✅ Successfully uploaded <strong>{uploaded_file.name}</strong>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Failed to load {uploaded_file.name}: {e}")

    if len(st.session_state['datasets']) > 0:
        st.markdown("""
            <div style="margin-top:20px; font-size:18px; font-weight:700; color:#6a0dad;">
                Select a dataset
            </div>
        """, unsafe_allow_html=True)

        default_index = 0
        if st.session_state['current_df_name'] and st.session_state['current_df_name'] in [d['name'] for d in st.session_state['datasets']]:
            default_index = [d['name'] for d in st.session_state['datasets']].index(st.session_state['current_df_name'])

        selected_dataset_name = st.selectbox(
            label="",
            options=[d['name'] for d in st.session_state['datasets']],
            index=default_index,
            key="upload_dataset_selector",
            label_visibility="collapsed"
        )

        current_df = next(
            (d['dataframe'] for d in st.session_state['datasets'] if d['name'] == selected_dataset_name),
            None
        )
        st.session_state['current_df'] = current_df
        st.session_state['current_df_name'] = selected_dataset_name

        st.markdown(f"""<div style="margin-top:20px; font-size:16px; color:#333;"><b>Showing preview for:</b> {selected_dataset_name}</div>""", unsafe_allow_html=True)

        if current_df is not None:
            st.markdown(current_df.head().to_html(classes="styled-table", index=False), unsafe_allow_html=True)

    else:
        st.markdown("""
            <div style="
                background-color: #e2e3ff;
                color: #6a0dad;
                padding: 15px 20px;
                border-left: 6px solid #6a0dad;
                border-radius: 8px;
                font-size: 16px;
                margin-top: 20px;
            ">
                Please upload one or more CSV files to begin.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- HOW IT WORKS ----------
elif st.session_state['view'] == 'how_it_works':
    how_html = """
    <style>
    .how-container {
        padding: 40px 8%;
        font-family: "Segoe UI", sans-serif;
        color: black;
    }
    .how-title {
        font-size: 42px;
        font-weight: 800;
        color: #6a0dad;
        margin-bottom: 20px;
    }
    .how-subtitle {
        font-size: 20px;
        color: #444;
        margin-bottom: 40px;
    }
    .step {
        margin-bottom: 40px;
        display: flex;
        align-items: flex-start;
    }
    .step-icon {
        font-size: 20px;
        background-color: #4b6cb7;
        color: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 20px;
        flex-shrink: 0;
    }
    .step-content h4 {
        margin: 0;
        font-size: 22px;
        color: #222;
    }
    .step-content p {
        font-size: 16px;
        color: #555;
        margin-top: 6px;
    }
    </style>

    <div class="how-container">
        <div class="how-title">How It Works</div>
        <div class="how-subtitle">
            Follow these simple steps to clean, explore, and download your data using our Smart Data Cleaning Assistant — <b>AI WrangleBot</b>.
        </div>

        <div class="step">
            <div class="step-icon">1</div>
            <div class="step-content">
                <h4>Upload Your Dataset</h4>
                <p>Choose a CSV file. Our assistant will instantly load and preview your data.</p>
            </div>
        </div>

        <div class="step">
            <div class="step-icon">2</div>
            <div class="step-content">
                <h4>Explore Your Data</h4>
                <p>View missing values, detect duplicates, and analyze column types.</p>
            </div>
        </div>

        <div class="step">
            <div class="step-icon">3</div>
            <div class="step-content">
                <h4>Clean with AI Guidance</h4>
                <p>Apply smart fixes like filling, dropping, and converting — guided by our AI engine with personalized suggestions and reasoning for each step.</p>
            </div>
        </div>

        <div class="step">
            <div class="step-icon">4</div>
            <div class="step-content">
                <h4>Download Cleaned Dataset</h4>
                <p>Download the final dataset for modeling, visualization, or reporting.</p>
            </div>
        </div>
    </div>
    """
    components.html(how_html, height=800, scrolling=True)


# ---------- EXPLORE MENU (Purple Theme Variant) ----------
elif st.session_state['view'] == 'explore':
    st.markdown("""
        <style>
        .explore-section {
            background-color: #e2e3ff;
            padding: 60px;
            margin: 50px 8%;
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            font-family: "Segoe UI", sans-serif;
        }
        .explore-title {
            font-size: 36px;
            font-weight: 800;
            color: #6a0dad;
            margin-bottom: 20px;
        }
        .explore-subtext {
            font-size: 18px;
            color: #222222;
            margin-bottom: 30px;
        }
        .section-heading {
            font-size: 24px;
            font-weight: bold;
            color: #6a0dad;
            margin-top: 40px;
            margin-bottom: 10px;
        }
        .styled-table th {
            background-color: #6a0dad;
            color: white;
            padding: 10px;
        }
        .styled-table td {
            padding: 10px;
            border: 1px solid #ddd;
            background-color: #f5f2ff;
            color: #222222;
        }
        .styled-table tr:nth-child(even) {
            background-color: #ece6ff;
        }
        .styled-table tr:hover {
            background-color: #e0dbff;
        }
        .label-purple {
            font-size: 18px;
            font-weight: 700;
            color: #222 !important;
            margin-bottom: 8px;
        }
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #6a0dad !important;
            border-radius: 8px !important;
            border: 1px solid #ccc !important;
        }
        div[data-baseweb="select"] div[role="button"] {
            color: #6a0dad !important;
            font-weight: 600;
        }
        ul[role="listbox"] {
            background-color: #ffffff !important;
            color: #000 !important;
            font-size: 16px;
        }
        li[role="option"]:hover {
            background-color: #e2e3ff !important;
        }
        </style>
        <div class="explore-section">
            <div class="explore-title">Explore Dataset</div>
            <div class="explore-subtext">Gain insights on structure, null values, duplicates, outliers, and more before cleaning.</div>
    """, unsafe_allow_html=True)

    if 'datasets' in st.session_state and len(st.session_state['datasets']) > 0:
        st.markdown('<div class="label-purple">Select a dataset to explore</div>', unsafe_allow_html=True)

        # Find the index of the current_df_name if it exists, otherwise default to 0
        default_index = 0
        if st.session_state['current_df_name'] and st.session_state['current_df_name'] in [d['name'] for d in st.session_state['datasets']]:
            default_index = [d['name'] for d in st.session_state['datasets']].index(st.session_state['current_df_name'])

        selected_dataset_name = st.selectbox(
            "",
            [d['name'] for d in st.session_state['datasets']],
            index=default_index, # Set default to previously selected or first
            key="explore_selectbox",
            label_visibility="collapsed"
        )

        current_df = next(
            (d['dataframe'] for d in st.session_state['datasets'] if d['name'] == selected_dataset_name),
            None
        )
        st.session_state['current_df'] = current_df # Update current_df in session state
        st.session_state['current_df_name'] = selected_dataset_name # Update current_df_name

        if current_df is not None:
            st.markdown(f"""<div style="margin-top:10px; font-size:16px; color:#222;"><b>Previewing:</b> {selected_dataset_name}</div>""", unsafe_allow_html=True)

            st.markdown(f"<div class='section-heading'>Dataset Preview</div>", unsafe_allow_html=True)
            st.markdown(current_df.head().to_html(classes="styled-table", index=False), unsafe_allow_html=True)

            st.markdown(f"<div class='section-heading'>Dataset Overview</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='label-purple'>Shape: {current_df.shape[0]} rows × {current_df.shape[1]} columns</div>", unsafe_allow_html=True)

            st.markdown(f"<div class='section-heading'>Column Data Types</div>", unsafe_allow_html=True)
            st.markdown(current_df.dtypes.rename('dtype').reset_index().rename(columns={'index': 'column'}).to_html(classes="styled-table", index=False), unsafe_allow_html=True)

            st.markdown(f"<div class='section-heading'>Null Values per Column</div>", unsafe_allow_html=True)
            st.markdown(current_df.isnull().sum().rename('nulls').reset_index().rename(columns={'index': 'column'}).to_html(classes="styled-table", index=False), unsafe_allow_html=True)

            # Outliers using IQR
            numeric_cols = current_df.select_dtypes(include=np.number).columns
            outlier_summary = []
            for col in numeric_cols:
                Q1 = current_df[col].quantile(0.25)
                Q3 = current_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = current_df[(current_df[col] < lower) | (current_df[col] > upper)]
                outlier_summary.append({
                    'Column': col,
                    'Outlier Count': outliers.shape[0]
                })
            outlier_df = pd.DataFrame(outlier_summary)
            st.markdown(f"<div class='section-heading'>Outliers Detected using IQR</div>", unsafe_allow_html=True)
            st.markdown(outlier_df.to_html(classes="styled-table", index=False), unsafe_allow_html=True)

            st.markdown(f"<div class='section-heading'>Duplicate Rows</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='label-purple'>Number of duplicate rows: {current_df.duplicated().sum()}</div>", unsafe_allow_html=True)

            st.markdown(f"<div class='section-heading'>Unique Values per Column</div>", unsafe_allow_html=True)
            st.markdown(current_df.nunique().rename('unique_values').reset_index().rename(columns={'index': 'column'}).to_html(classes="styled-table", index=False), unsafe_allow_html=True)

            st.markdown(f"<div class='section-heading'>Show Value Counts</div>", unsafe_allow_html=True)
            st.markdown('<div class="label-purple">Select a column to show value counts (optional):</div>', unsafe_allow_html=True)
            selected_col = st.selectbox(
                "",
                [""] + list(current_df.columns),
                key="value_counts_col",
                label_visibility="collapsed"
            )

            if selected_col != "":
                vc_df = current_df[selected_col].value_counts().reset_index()
                vc_df.columns = ["Value", "Count"]
                st.markdown(f"<div style='font-weight:600; color:#6a0dad;'>Value counts for column: {selected_col}</div>", unsafe_allow_html=True)
                st.markdown(vc_df.to_html(classes="styled-table", index=False), unsafe_allow_html=True)
        else:
            st.warning("No dataset selected.")
    else:
        st.markdown("""
            <div style="
                background-color: #e2e3ff;
                color: #6a0dad;
                padding: 15px 20px;
                border-left: 6px solid #6a0dad;
                border-radius: 8px;
                font-size: 16px;
                margin-top: 20px;
            ">
                Please upload one or more CSV files to explore.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state['view'] == 'clean_data':
    
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    def show_static_message(msg_type, msg_text):
        color_map = {
            "success": "#d4edda",
            "error": "#f8d7da",
            "info": "#cce5ff",
            "warning": "#fff3cd"
        }
        text_color = {
            "success": "#155724",
            "error": "#721c24",
            "info": "#004085",
            "warning": "#856404"
        }
        return f"""
            <div style='
                background-color: {color_map[msg_type]};
                color: {text_color[msg_type]};
                padding: 8px 16px;
                border-radius: 6px;
                margin: 10px 0;
                font-size: 14px;
            '>{msg_text}</div>
        """

    st.markdown("""
<style>
.clean-section {
    background-color: #f6edff;
    padding: 60px;
    margin: 50px 8%;
    border-radius: 20px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    font-family: "Segoe UI", sans-serif;
}
.clean-title {
    font-size: 36px;
    font-weight: 800;
    color: #6a0dad;
    margin-bottom: 20px;
}
.clean-subtext {
    font-size: 18px;
    color: #222222;
    margin-bottom: 30px;
}
.label-purple {
    font-size: 18px;
    font-weight: 700;
    color: #000 !important;
    margin-bottom: 8px;
}
.subheading-purple {
    font-size: 24px;
    font-weight: 700;
    color: #6a0dad;
    margin-top: 40px;
}
.subtext-black {
    font-size: 16px;
    color: #000 !important;
    margin-bottom: 15px;
}
.ai-suggestion {
    background-color: #f0f8ff;
    border-left: 4px solid #4682b4;
    padding: 12px;
    margin: 10px 0;
    border-radius: 4px;
    font-size: 14px;
    color: #000000 !important;
}

/* Set all labels and placeholders to black */
label, .stSelectbox label, .stMultiselect label, .stRadio label,
.stRadio label span, .stRadio > div > div,
.css-1wa3eu0-placeholder, .css-14el2xx, .css-1okebmr-indicatorSeparator {
    color: #000 !important;
}

/* Fix radio options (like Z-score, Min-Max) */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label span {
    color: #000 !important;
}

/* Dropdown and multiselect styling */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000 !important;
    border-radius: 8px !important;
    border: 1px solid #ccc !important;
}
div[data-baseweb="select"] div[role="button"] {
    color: #000 !important;
    font-weight: 600;
}
ul[role="listbox"] {
    background-color: #ffffff !important;
    color: #000 !important;
    font-size: 16px;
}
li[role="option"]:hover {
    background-color: #e2e3ff !important;
}

/* Purple styled buttons */
.purple-button button,
button[kind="secondary"] {
    background-color: #6a0dad !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600;
    border-radius: 8px !important;
    padding: 6px 12px;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] button {
    background-color: #3b0a4e !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] button:focus,
section[data-testid="stSidebar"] button:hover {
    outline: none !important;
    border: none !important;
    box-shadow: none !important;
}
</style>

<div class="clean-section">
    <div class="clean-title">⚙ Clean Your Data</div>
    <div class="clean-subtext">Choose a dataset and customize the cleaning options per column using AI WrangleBot.</div>
""", unsafe_allow_html=True)

    # ---------- MAIN FUNCTIONALITY ----------
    if 'datasets' in st.session_state and len(st.session_state['datasets']) > 0:
        st.markdown('<div class="label-purple">Select a dataset to clean</div>', unsafe_allow_html=True)
        selected_dataset_name = st.selectbox(
            "",
            [d['name'] for d in st.session_state['datasets']],
            key="clean_selectbox",
            label_visibility="collapsed"
        )

        current_df = next((d['dataframe'] for d in st.session_state['datasets'] if d['name'] == selected_dataset_name), None)
        st.session_state['current_df'] = current_df
        st.session_state['current_df_name'] = selected_dataset_name

        if current_df is not None:
            df = current_df.copy()
            st.markdown(f"<div class='subtext-black'><b>Selected Dataset:</b> {selected_dataset_name}</div>", unsafe_allow_html=True)

            # Remove Duplicates
            st.markdown("<div class='subheading-purple'>Remove Duplicate Rows</div>", unsafe_allow_html=True)
            dup_count = df.duplicated().sum()
            st.markdown(f"<div class='subtext-black'>Found {dup_count} duplicate rows.</div>", unsafe_allow_html=True)
            
            # AI Suggestion for Duplicates
            if st.button("Get AI Suggestion for Duplicates"):
                suggestion = get_duplicate_suggestion(df, selected_dataset_name)
                st.session_state['duplicate_suggestion'] = suggestion
            if 'duplicate_suggestion' in st.session_state:
                st.markdown(f"""
                    <div class="ai-suggestion">
                        <strong>🤖 AI Suggestion:</strong><br>
                        {st.session_state['duplicate_suggestion']}
                    </div>
                """, unsafe_allow_html=True)
            
            if st.button("Remove Duplicates"):
                df = df.drop_duplicates()
                st.session_state['duplicate_msg'] = show_static_message("success", "✅ Duplicate rows removed.")
            if 'duplicate_msg' in st.session_state:
                st.markdown(st.session_state['duplicate_msg'], unsafe_allow_html=True)

            st.divider()

            # Handle Missing Values
            st.markdown("<div class='subheading-purple'>Handle Missing Values (Column-wise)</div>", unsafe_allow_html=True)
            if 'cleaned_missing_msgs' not in st.session_state:
                st.session_state['cleaned_missing_msgs'] = {}
            
            if 'missing_suggestions' not in st.session_state:
                st.session_state['missing_suggestions'] = {}

            for col in df.columns[df.isnull().any()]:
                st.markdown(f"<div class='subtext-black'><b>{col}</b> → {df[col].isnull().sum()} missing values</div>", unsafe_allow_html=True)
                
                # AI Suggestion for Missing Values
                if st.button(f"Get AI Suggestion for {col}", key=f"ai_missing_{col}"):
                    suggestion = get_missing_value_suggestion(df, col, selected_dataset_name)
                    st.session_state['missing_suggestions'][col] = suggestion
                
                if col in st.session_state['missing_suggestions']:
                    st.markdown(f"""
                        <div class="ai-suggestion">
                            <strong>🤖 AI Suggestion for {col}:</strong><br>
                            {st.session_state['missing_suggestions'][col]}
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"<div class='label-purple'>Select strategy for {col}:</div>", unsafe_allow_html=True)
                strategy = st.selectbox(
                    "",
                    ["Do Nothing", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Interpolate"],
                    key=f"strategy_{selected_dataset_name}_{col}",
                    label_visibility="collapsed"
                )
                if st.button(f"Apply to {col}", key=f"apply_{selected_dataset_name}_{col}"):
                    try:
                        msg = ""
                        if strategy == "Drop Rows":
                            df = df[df[col].notnull()]
                            msg = show_static_message("success", f"✅ Rows with missing {col} dropped.")
                        elif strategy == "Fill with Mean":
                            if pd.api.types.is_numeric_dtype(df[col]):
                                df[col].fillna(df[col].mean(), inplace=True)
                                msg = show_static_message("success", f"✅ Filled missing {col} with mean.")
                            else:
                                msg = show_static_message("warning", f"⚠ Cannot fill mean for non-numeric column: {col}")
                        elif strategy == "Fill with Median":
                            if pd.api.types.is_numeric_dtype(df[col]):
                                df[col].fillna(df[col].median(), inplace=True)
                                msg = show_static_message("success", f"✅ Filled missing {col} with median.")
                            else:
                                msg = show_static_message("warning", f"⚠ Cannot fill median for non-numeric column: {col}")
                        elif strategy == "Fill with Mode":
                            mode_val = df[col].mode()
                            if not mode_val.empty:
                                df[col].fillna(mode_val[0], inplace=True)
                                msg = show_static_message("success", f"✅ Filled missing {col} with mode.")
                            else:
                                msg = show_static_message("warning", f"⚠ No mode found for {col}")
                        elif strategy == "Interpolate":
                            if pd.api.types.is_numeric_dtype(df[col]):
                                df[col] = df[col].interpolate()
                                msg = show_static_message("success", f"✅ Interpolated missing values in {col}.")
                            else:
                                msg = show_static_message("warning", f"⚠ Cannot interpolate non-numeric column: {col}")
                        elif strategy == "Do Nothing":
                            msg = show_static_message("info", f"ℹ Skipped {col} (no action taken).")
                        st.session_state['cleaned_missing_msgs'][col] = msg
                    except Exception as e:
                        st.session_state['cleaned_missing_msgs'][col] = show_static_message("error", f"❌ Error: {e}")

                if col in st.session_state['cleaned_missing_msgs']:
                    st.markdown(st.session_state['cleaned_missing_msgs'][col], unsafe_allow_html=True)

            st.divider()

            # Convert Data Types
            st.markdown("<div class='subheading-purple'>Convert Column Data Types</div>", unsafe_allow_html=True)
            if 'convert_msgs' not in st.session_state:
                st.session_state['convert_msgs'] = {}
            
            if 'dtype_suggestions' not in st.session_state:
                st.session_state['dtype_suggestions'] = {}

            for col in df.columns:
                current_dtype = df[col].dtype
                st.markdown(f"<div class='subtext-black'><b>{col}</b> → current type: {current_dtype}</div>", unsafe_allow_html=True)
                
                # AI Suggestion for Data Type Conversion
                if st.button(f"Get AI Suggestion for {col}", key=f"ai_dtype_{col}"):
                    suggestion = get_dtype_conversion_suggestion(df, col, selected_dataset_name)
                    st.session_state['dtype_suggestions'][col] = suggestion
                
                if col in st.session_state['dtype_suggestions']:
                    st.markdown(f"""
                        <div class="ai-suggestion">
                            <strong>🤖 AI Suggestion for {col}:</strong><br>
                            {st.session_state['dtype_suggestions'][col]}
                        </div>
                    """, unsafe_allow_html=True)
                
                new_type = st.selectbox(
                    "",
                    ["No Change", "string", "int", "float", "datetime"],
                    key=f"dtype_{col}",
                    label_visibility="collapsed"
                )
                if st.button(f"Convert {col} to {new_type}", key=f"convert_{col}"):
                    try:
                        if new_type != "No Change":
                            if new_type == "datetime":
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                            else:
                                df[col] = df[col].astype(new_type)
                            msg = show_static_message("success", f"✅ Converted {col} to {new_type}.")
                        else:
                            msg = show_static_message("info", f"ℹ No conversion applied to {col}.")
                        st.session_state['convert_msgs'][col] = msg
                    except Exception as e:
                        st.session_state['convert_msgs'][col] = show_static_message("error", f"❌ Failed to convert {col}: {e}")
                if col in st.session_state['convert_msgs']:
                    st.markdown(st.session_state['convert_msgs'][col], unsafe_allow_html=True)

            st.divider()

            # Drop Columns
            st.markdown("<div class='subheading-purple'>Drop Unwanted Columns</div>", unsafe_allow_html=True)
            drop_cols = st.multiselect("Select columns to drop", df.columns.tolist(), key="drop_cols")
            if st.button("Drop Selected Columns"):
                df.drop(columns=drop_cols, inplace=True)
                st.session_state['drop_msg'] = show_static_message("success", f"✅ Dropped columns: {', '.join(drop_cols)}")
            if 'drop_msg' in st.session_state:
                st.markdown(st.session_state['drop_msg'], unsafe_allow_html=True)

            st.divider()

            # Outlier Filtering
            st.markdown("<div class='subheading-purple'>Filter Outliers and Anomalies</div>", unsafe_allow_html=True)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_outlier_cols = st.multiselect("Select numeric columns to filter outliers using IQR", num_cols, key="outlier_cols")
            
            # AI Suggestion for Outliers
            if st.button("Get AI Suggestion for Outliers"):
                suggestion = get_outlier_suggestion(df, num_cols, selected_dataset_name)
                st.session_state['outlier_suggestion'] = suggestion
            if 'outlier_suggestion' in st.session_state:
                st.markdown(f"""
                    <div class="ai-suggestion">
                        <strong>🤖 AI Suggestion:</strong><br>
                        {st.session_state['outlier_suggestion']}
                    </div>
                """, unsafe_allow_html=True)
            
            if st.button("Remove Outliers"):
                try:
                    for col in selected_outlier_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        df = df[(df[col] >= lower) & (df[col] <= upper)]
                    st.session_state['outlier_msg'] = show_static_message("success", f"✅ Outliers removed from: {', '.join(selected_outlier_cols)}")
                except Exception as e:
                    st.session_state['outlier_msg'] = show_static_message("error", f"❌ Failed to remove outliers: {e}")
            if 'outlier_msg' in st.session_state:
                st.markdown(st.session_state['outlier_msg'], unsafe_allow_html=True)

            st.divider()

            # Scaling
            st.markdown("<div class='subheading-purple'>Standardize and Normalize Data</div>", unsafe_allow_html=True)
            selected_norm_cols = st.multiselect("Select numeric columns to scale", num_cols, key="scale_cols")
            
            # AI Suggestion for Scaling
            if st.button("Get AI Suggestion for Scaling"):
                suggestion = get_scaling_suggestion(df, num_cols, selected_dataset_name)
                st.session_state['scaling_suggestion'] = suggestion
            if 'scaling_suggestion' in st.session_state:
                st.markdown(f"""
                    <div class="ai-suggestion">
                        <strong>🤖 AI Suggestion:</strong><br>
                        {st.session_state['scaling_suggestion']}
                    </div>
                """, unsafe_allow_html=True)
            
            scale_type = st.radio("Select scaling method", ["Standardization (Z-score)", "Normalization (Min-Max)"], key="scaling_method")
            if st.button("Apply Scaling"):
                try:
                    scaler = StandardScaler() if scale_type == "Standardization (Z-score)" else MinMaxScaler()
                    df[selected_norm_cols] = scaler.fit_transform(df[selected_norm_cols])
                    st.session_state['scaling_msg'] = show_static_message("success", f"✅ Applied {scale_type} to: {', '.join(selected_norm_cols)}")
                except Exception as e:
                    st.session_state['scaling_msg'] = show_static_message("error", f"❌ Failed to apply scaling: {e}")
            if 'scaling_msg' in st.session_state:
                st.markdown(st.session_state['scaling_msg'], unsafe_allow_html=True)

            # Save Cleaned Data
            st.session_state['cleaned_df'] = df

        else:
            st.markdown(show_static_message("warning", "⚠ No dataset selected."), unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="
                background-color: #e2e3ff;
                color: #6a0dad;
                padding: 15px 20px;
                border-left: 6px solid #6a0dad;
                border-radius: 8px;
                font-size: 16px;
                margin-top: 20px;
            ">
                Please upload one or more CSV files to clean.
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
# ---------- DOWNLOAD ----------
elif st.session_state['view'] == 'download':
    st.markdown("""
        <style>
        .download-section {
            background-color: #e2e3ff;
            padding: 60px;
            margin: 50px 8%;
            border-radius: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
            font-family: "Segoe UI", sans-serif;
        }
        .download-title {
            font-size: 36px;
            font-weight: 800;
            color: #6a0dad;
            margin-bottom: 20px;
        }
        .download-subtext {
            font-size: 18px;
            color: #222222;
            margin-bottom: 30px;
        }
        .label-purple {
            font-size: 18px;
            font-weight: 700;
            color: #222 !important;
            margin-bottom: 8px;
        }
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #6a0dad !important;
            border-radius: 8px !important;
            border: 1px solid #ccc !important;
        }
        div[data-baseweb="select"] div[role="button"] {
            color: #6a0dad !important;
            font-weight: 600;
        }
        ul[role="listbox"] {
            background-color: #ffffff !important;
            color: #000 !important;
            font-size: 16px;
        }
        li[role="option"]:hover {
            background-color: #e2e3ff !important;
        }
        .styled-table th {
            background-color: #6a0dad;
            color: white;
            padding: 10px;
        }
        .styled-table td {
            padding: 10px;
            border: 1px solid #ddd;
            background-color: #f5f2ff;
            color: #222222;
        }
        .styled-table tr:nth-child(even) {
            background-color: #ece6ff;
        }
        .styled-table tr:hover {
            background-color: #e0dbff;
        }
        </style>
        <div class="download-section">
            <div class="download-title">Download Cleaned Data</div>
            <div class="download-subtext">Download your cleaned datasets in CSV format.</div>
    """, unsafe_allow_html=True)

    if st.session_state['cleaned_dfs']:
        st.markdown('<div class="label-purple">Select dataset to download</div>', unsafe_allow_html=True)
        
        selected_download = st.selectbox(
            "",
            list(st.session_state['cleaned_dfs'].keys()),
            key="download_select",
            label_visibility="collapsed"
        )

        if selected_download:
            cleaned_df = st.session_state['cleaned_dfs'][selected_download]
            
            st.markdown(f"<div class='label-purple'>Preview of {selected_download} (cleaned)</div>", unsafe_allow_html=True)
            st.markdown(cleaned_df.head().to_html(classes="styled-table", index=False), unsafe_allow_html=True)
            
            # Download button
            csv = cleaned_df.to_csv(index=False)
            st.download_button(
                label="Download Cleaned CSV",
                data=csv,
                file_name=f"cleaned_{selected_download}",
                mime="text/csv"
            )

    else:
        st.markdown(show_static_message("warning", "No cleaned datasets available. Please upload and clean data first."), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- ABOUT ----------
elif st.session_state['view'] == 'about':
    about_html = """
    <style>
    .about-container {
        background: #e2e3ff;
        padding: 60px 8%;
        border-radius: 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        font-family: "Segoe UI", sans-serif;
        margin: 30px 5%;
    }
    .about-title {
        font-size: 42px;
        font-weight: 800;
        color: #4b6cb7;
        margin-bottom: 20px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .about-subtitle {
        font-size: 20px;
        color: #555;
        margin-bottom: 40px;
        line-height: 1.6;
    }
    .credit-list {
        font-size: 18px;
        color: #333;
        line-height: 1.8;
        margin-left: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #666;
        font-size: 15px;
    }
    </style>

    <div class="about-container">
        <div class="about-title">About & Credits</div>
        
        <div class="about-subtitle">
            The Smart Data Cleaning Assistant is a project developed as part of a professional initiative to simplify and automate the data preprocessing workflow.
        </div>

        <div class="credit-list">
            <strong>Developed by:</strong>
            <ul>
                <li>👤 Minahil Azeem</li>
                <li>👤 Ahmed Baig</li>
                <li>👤 Abdullah Ahmed</li>
            </ul>
        </div>

        <div class="footer">
            © 2025 Smart Data Cleaning Assistant | All rights reserved
        </div>
    </div>
    """
    components.html(about_html, height=500, scrolling=False)