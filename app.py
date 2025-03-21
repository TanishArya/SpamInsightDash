import streamlit as st
import pandas as pd
import base64
from io import StringIO
import os
import time

from util import load_sample_data, preprocess_data, get_summary_stats
from text_analysis import extract_word_frequencies, get_message_length_stats, analyze_common_words
from visualization import (
    plot_spam_distribution, 
    plot_word_frequency, 
    plot_message_length_comparison,
    plot_character_count_distribution,
    plot_word_clouds,
    plot_confusion_matrix
)

# Page configuration
st.set_page_config(
    page_title="Spam Detection Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("📊 Spam Detection Visualization Dashboard")
st.markdown("""
This dashboard provides interactive visualizations and metrics for analyzing spam detection patterns.
Upload your spam detection dataset (CSV format) or use our sample data to get started.
""")

# Sidebar
with st.sidebar:
    st.header("Controls")
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source",
        ["Upload your own data", "Use sample data"],
        index=1
    )
    
    uploaded_file = None
    if data_source == "Upload your own data":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        st.markdown("""
        ### Expected CSV Format:
        - Column with spam/not spam labels
        - Column with message text
        """)
        
        if uploaded_file is None:
            st.warning("Please upload a CSV file or select 'Use sample data'")
    
    # Filter options
    st.subheader("Filters")
    
    # These will be populated once we have data
    message_length_filter = st.slider(
        "Filter by Message Length",
        min_value=0,
        max_value=500,
        value=(0, 500),
        disabled=(data_source == "Upload your own data" and uploaded_file is None)
    )
    
    st.markdown("---")
    
    # Export options
    st.subheader("Export Options")
    export_format = st.selectbox(
        "Select Export Format",
        ["CSV", "Excel"],
        disabled=(data_source == "Upload your own data" and uploaded_file is None)
    )

# Load data
data = None
load_state = st.text("Loading data...")

try:
    if data_source == "Use sample data" or (data_source == "Upload your own data" and uploaded_file is not None):
        if data_source == "Use sample data":
            data = load_sample_data()
        else:
            data = pd.read_csv(uploaded_file)
            
        # Preprocess data
        data = preprocess_data(data)
        
        # Apply filters
        data_filtered = data[
            (data['message_length'] >= message_length_filter[0]) &
            (data['message_length'] <= message_length_filter[1])
        ]
        
        load_state.text("Data loaded successfully!")
        time.sleep(0.5)
        load_state.empty()
    
except Exception as e:
    load_state.error(f"Error loading data: {str(e)}")
    st.stop()

if data is not None:
    # Dashboard Layout
    # First row - Summary metrics
    st.header("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    summary_stats = get_summary_stats(data)
    
    with col1:
        st.metric("Total Messages", summary_stats['total_messages'])
    
    with col2:
        st.metric("Spam Messages", summary_stats['spam_count'], 
                  f"{summary_stats['spam_percentage']:.1f}%")
    
    with col3:
        st.metric("Non-Spam Messages", summary_stats['non_spam_count'],
                 f"{summary_stats['non_spam_percentage']:.1f}%")
    
    with col4:
        st.metric("Avg. Message Length", f"{summary_stats['avg_message_length']:.1f} chars",
                 f"{summary_stats['avg_spam_length'] - summary_stats['avg_non_spam_length']:.1f} chars (spam-non spam)")
    
    # Metrics tabs
    st.markdown("---")
    
    tabs = st.tabs(["Classification Metrics", "Content Analysis", "Word Frequency", "Length Analysis"])
    
    with tabs[0]:
        st.subheader("Spam vs. Non-Spam Distribution")
        st.plotly_chart(plot_spam_distribution(data), use_container_width=True)
        
        st.subheader("Confusion Matrix Visualization")
        st.caption("Visualization of true positive, true negative, false positive, and false negative predictions")
        st.plotly_chart(plot_confusion_matrix(data), use_container_width=True)
    
    with tabs[1]:
        st.subheader("Word Clouds by Classification")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Spam Word Cloud")
            fig_spam_cloud = plot_word_clouds(data, message_type='spam')
            st.pyplot(fig_spam_cloud)
        
        with col2:
            st.subheader("Non-Spam Word Cloud")
            fig_non_spam_cloud = plot_word_clouds(data, message_type='non_spam')
            st.pyplot(fig_non_spam_cloud)
        
        st.subheader("Common Words Analysis")
        common_words_analysis = analyze_common_words(data)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Words in Spam Messages")
            st.table(common_words_analysis['spam_words'])
        
        with col2:
            st.subheader("Top Words in Non-Spam Messages")
            st.table(common_words_analysis['non_spam_words'])
    
    with tabs[2]:
        st.subheader("Word Frequency Distribution")
        word_frequencies = extract_word_frequencies(data)
        st.plotly_chart(plot_word_frequency(word_frequencies), use_container_width=True)
    
    with tabs[3]:
        st.subheader("Message Length Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Message Length Comparison")
            st.plotly_chart(plot_message_length_comparison(data), use_container_width=True)
        
        with col2:
            st.subheader("Character Count Distribution")
            st.plotly_chart(plot_character_count_distribution(data), use_container_width=True)
    
    
    # Export section
    st.markdown("---")
    st.subheader("Export Data")
    
    export_col1, export_col2 = st.columns([3, 1])
    
    with export_col1:
        export_options = st.multiselect(
            "Select data to export",
            ["Filtered Data", "Word Frequencies", "Length Statistics"],
            default=["Filtered Data"]
        )
    
    with export_col2:
        if st.button("Export Selected Data"):
            if "Filtered Data" in export_options:
                if export_format == "CSV":
                    csv = data_filtered.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="spam_detection_data.csv">Download Filtered Data (CSV)</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:  # Excel
                    output = StringIO()
                    data_filtered.to_excel(output, index=False)
                    b64 = base64.b64encode(output.getvalue().encode()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="spam_detection_data.xlsx">Download Filtered Data (Excel)</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("Spam Detection Visualization Dashboard | Powered by Streamlit")
else:
    # Show message when no data is loaded
    if data_source == "Upload your own data" and uploaded_file is None:
        st.info("Please upload a CSV file or select 'Use sample data' to view the dashboard.")
