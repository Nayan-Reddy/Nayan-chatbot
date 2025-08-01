import streamlit as st
import gspread
import pandas as pd
import plotly.express as px
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Chatbot Analytics",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data(ttl=600)  # Cache the data for 10 minutes
def load_data():
    """Loads interaction data from the Google Sheet."""
    try:
        # Authenticate using Streamlit's secrets
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        # Open the sheet by its name
        spreadsheet = gc.open("Chatbot Logs")
        # Select the first worksheet
        worksheet = spreadsheet.sheet1
        # Get all records from the sheet
        records = worksheet.get_all_records()
        # Convert to a Pandas DataFrame
        df = pd.DataFrame(records)
        # Convert timestamp column to datetime objects
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except gspread.exceptions.SpreadsheetNotFound:
        st.error("Spreadsheet 'Chatbot Logs' not found. Please check the name and sharing permissions.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        return pd.DataFrame()

df = load_data()

# --- Main Dashboard ---
st.title("🤖 Chatbot Analytics Dashboard")

if df.empty:
    st.warning("No data available to display. Use the chatbot to generate logs.")
else:
    # --- KPIs ---
    st.header("Key Performance Indicators")
    total_users = df['session_id'].nunique()
    total_questions = len(df)
    avg_questions_per_user = total_questions / total_users if total_users > 0 else 0
    avg_response_time = df['response_time_ms'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users (Sessions)", f"{total_users}")
    col2.metric("Total Questions Asked", f"{total_questions}")
    col3.metric("Avg Qs / User", f"{avg_questions_per_user:.2f}")
    col4.metric("Avg Response Time (ms)", f"{avg_response_time:.0f}")

    st.divider()

    # --- Visualizations ---
    col1, col2 = st.columns(2)

    with col1:
        st.header("Response Sources")
        source_counts = df['response_source'].value_counts()
        fig_source = px.pie(
            source_counts,
            values=source_counts.values,
            names=source_counts.index,
            title="Distribution of Answer Sources",
            hole=0.3
        )
        st.plotly_chart(fig_source, use_container_width=True)

    with col2:
        st.header("Usage Over Time")
        df['date'] = df['timestamp'].dt.date
        daily_usage = df.groupby('date').size().reset_index(name='count')
        fig_usage = px.bar(
            daily_usage,
            x='date',
            y='count',
            title="Questions Asked Per Day"
        )
        st.plotly_chart(fig_usage, use_container_width=True)

    st.divider()

    # --- Most Asked Questions ---
    st.header("Most Frequent Questions")
    top_questions = df['user_query'].str.lower().str.strip().value_counts().head(15)
    st.dataframe(top_questions, use_container_width=True, height=500)

    # --- Raw Data Explorer ---
    st.divider()
    with st.expander("Explore Raw Interaction Logs"):
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)