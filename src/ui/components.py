"""
Reusable UI Components for TReA Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
from datetime import datetime

from ..config import settings


def display_header():
    """Display application header"""
    st.set_page_config(
        page_title=settings.page_title,
        page_icon=settings.page_icon,
        layout=settings.layout,
        initial_sidebar_state="expanded"
    )
    
    st.title(f"{settings.page_icon} {settings.app_name}")
    st.markdown(f"""
    **Version:** {settings.app_version}  
    **Treasury Document Processing & Journal Mapping**
    """)
    st.divider()


def display_sidebar():
    """Display sidebar with navigation and settings"""
    with st.sidebar:
        st.header("🔧 Settings")
        
        # API Configuration
        st.subheader("API Configuration")
        api_base_url = st.text_input(
            "API Base URL", 
            value=settings.api_base_url,
            help="Base URL for TReA backend API"
        )
        
        api_token = st.text_input(
            "API Token", 
            type="password",
            value=settings.api_token or "",
            help="Bearer token for API authentication"
        )
        
        # File Upload Settings
        st.subheader("File Upload")
        max_file_size = st.number_input(
            "Max File Size (MB)", 
            min_value=1, 
            max_value=100, 
            value=settings.max_file_size_mb
        )
        
        # Store settings in session state
        st.session_state.api_base_url = api_base_url
        st.session_state.api_token = api_token
        st.session_state.max_file_size_mb = max_file_size
        
        st.divider()
        
        # Navigation
        st.subheader("📊 Navigation")
        page = st.radio(
            "Select Page:",
            [
                "📄 Document Processing", 
                "📈 Analytics Dashboard", 
                "📋 Journal Suggestions",
                "⚙️ System Status",
                "📊 Vector Embeddings",
                "📋 Journal Setup",
                "📖 Definition Search"
            ],
            index=0
        )
        
        return page


def file_upload_component():
    """Enhanced multimodal file upload component with validation"""
    st.subheader("📁 Upload Treasury Document")
    
    # Add info about supported formats
    with st.expander("ℹ️ Supported File Formats", expanded=False):
        st.markdown("""
        **📄 PDF Files**: Treasury statements (DBS Singapore format)
        - Automatically processed through TReA backend API
        - Full PDF parsing and transaction extraction
        
        **📝 Text Files (.txt)**: Plain text transaction data
        - Format: `TRANSACTION_TYPE ASSET_CLASS AMOUNT CURRENCY DATE DESCRIPTION`
        - Example: `BUY STOCK 1000 USD 2024-01-15 Purchase of equity`
        
        **📊 JSON Files (.json)**: Structured transaction data
        ```json
        {
          "transactions": [
            {
              "transaction_type": "BUY",
              "asset_class": "STOCK", 
              "amount": 1000,
              "currency": "USD",
              "date": "2024-01-15",
              "description": "Purchase of equity"
            }
          ]
        }
        ```
        
        **📈 CSV Files (.csv)**: Tabular transaction data
        - Columns: transaction_type, asset_class, amount, currency, date, description
        - Headers are automatically mapped to standard format
        """)
    
    # Get allowed extensions without dots for streamlit
    allowed_extensions = [ext.lstrip('.') for ext in settings.allowed_file_types]
    
    uploaded_file = st.file_uploader(
        "Choose a treasury document file",
        type=allowed_extensions,
        help=f"Upload a treasury document (PDF, TXT, JSON, or CSV) - max {settings.max_file_size_mb}MB"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{file_size_mb:.2f} MB")
        with col3:
            st.metric("File Type", uploaded_file.type or f".{file_extension}")
        with col4:
            # Show processing method based on file type
            if file_extension == 'pdf':
                processing_method = "🔧 TReA API"
            else:
                processing_method = "🤖 Direct Processing"
            st.metric("Processing", processing_method)
        
        # Show file type specific information
        if file_extension == 'pdf':
            st.info("📄 PDF files will be processed through the TReA backend API for full extraction and analysis.")
        elif file_extension in ['txt', 'json', 'csv']:
            st.info("📝 Text-based files will be processed directly with AI enhancement and definition lookup.")
    
    return uploaded_file


def processing_status_component(status: str, message: str = ""):
    """Display processing status"""
    if status == "processing":
        st.info("🔄 Processing document...")
        if message:
            st.write(message)
    elif status == "success":
        st.success("✅ Document processed successfully!")
        if message:
            st.write(message)
    elif status == "error":
        st.error("❌ Processing failed")
        if message:
            st.write(f"Error: {message}")
    elif status == "warning":
        st.warning("⚠️ Warning")
        if message:
            st.write(message)


def transaction_summary_component(summary: Dict[str, Any]):
    """Display transaction summary metrics"""
    st.subheader("📊 Transaction Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Transactions", 
            summary.get("total_transactions", 0),
            help="Total number of transactions found"
        )
    
    with col2:
        st.metric(
            "Cash Transactions", 
            summary.get("cash_transactions", 0),
            help="Number of cash-related transactions"
        )
    
    with col3:
        st.metric(
            "Asset Transactions", 
            summary.get("asset_transactions", 0),
            help="Number of asset-related transactions"
        )
    
    with col4:
        st.metric(
            "Unique Pairs", 
            summary.get("unique_pairs", 0),
            help="Number of unique transaction type and asset class pairs"
        )


def transaction_types_chart(summary: Dict[str, Any]):
    """Display transaction types chart"""
    if not summary.get("transaction_types"):
        st.info("No transaction types data available")
        return
    
    st.subheader("📈 Transaction Types Distribution")
    
    # Count occurrences of each transaction type
    transaction_counts = {}
    for tx_type in summary["transaction_types"]:
        transaction_counts[tx_type] = transaction_counts.get(tx_type, 0) + 1
    
    # Create pie chart
    fig = px.pie(
        values=list(transaction_counts.values()),
        names=list(transaction_counts.keys()),
        title="Transaction Types Distribution"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, use_container_width=True)


def transactions_table_component(df: pd.DataFrame):
    """Display transactions table"""
    if df.empty:
        st.info("No transaction data available")
        return
    
    st.subheader("📋 Transaction Details")
    
    # Add filters
    col1, col2 = st.columns(2)
    
    with col1:
        transaction_filter = st.multiselect(
            "Filter by Transaction Type",
            options=df["transaction_type"].unique(),
            default=df["transaction_type"].unique()
        )
    
    with col2:
        category_filter = st.multiselect(
            "Filter by Category",
            options=df["category"].unique(),
            default=df["category"].unique()
        )
    
    # Apply filters
    filtered_df = df[
        (df["transaction_type"].isin(transaction_filter)) &
        (df["category"].isin(category_filter))
    ]
    
    # Display table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def system_status_component(api_client):
    """Display system status and health checks"""
    st.subheader("🔍 System Status")
    
    # API Health Check
    with st.spinner("Checking API health..."):
        health_status = api_client.check_health()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if health_status["success"]:
            st.success("✅ API Connection: Healthy")
            st.write(f"Status Code: {health_status['status_code']}")
        else:
            st.error("❌ API Connection: Failed")
            st.write(f"Error: {health_status.get('error', 'Unknown error')}")
    
    with col2:
        st.info("🔧 Configuration")
        st.write(f"**Base URL:** {api_client.base_url}")
        st.write(f"**Timeout:** {api_client.timeout}s")
        st.write(f"**Token Set:** {'Yes' if api_client.token else 'No'}")


def results_expander_component(results: Dict[str, Any]):
    """Display detailed results in expandable sections"""
    if not results.get("success"):
        return
    
    st.subheader("🔍 Detailed Results")
    
    # File Information
    with st.expander("📄 File Information"):
        file_info = results.get("file_info", {})
        if file_info:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Filename:** {file_info.get('filename', 'N/A')}")
                st.write(f"**Size:** {file_info.get('size_mb', 0):.2f} MB")
            with col2:
                st.write(f"**Extension:** {file_info.get('extension', 'N/A')}")
                st.write(f"**Size (bytes):** {file_info.get('size_bytes', 0):,}")
    
    # Extraction Data
    with st.expander("🔍 Extraction Data"):
        extraction_data = results.get("extraction_data", {})
        if extraction_data:
            st.json(extraction_data)
        else:
            st.info("No extraction data available")
    
    # Transformed Data
    with st.expander("🔄 Transformed Data"):
        transformed_data = results.get("transformed_data", {})
        if transformed_data:
            st.json(transformed_data)
        else:
            st.info("No transformed data available")
    
    # Mapped Data
    with st.expander("🗺️ Mapped Data"):
        mapped_data = results.get("mapped_data", {})
        if mapped_data:
            st.json(mapped_data)
        else:
            st.info("No mapped data available")


def error_display_component(error_message: str):
    """Display error message with helpful information"""
    st.error("❌ Processing Error")
    
    with st.expander("Error Details"):
        st.code(error_message)
        
        st.markdown("""
        **Troubleshooting Tips:**
        1. Check if the PDF file is valid and not corrupted
        2. Verify API connection settings in the sidebar
        3. Ensure the API token is valid and not expired
        4. Check if the file size is within limits
        5. Try uploading a different PDF file
        """)


def file_type_distribution_chart(file_type_counts):
    """Display file type distribution chart"""
    if len(file_type_counts) == 0:
        st.info("No data available for chart")
        return
    
    fig = px.pie(
        values=file_type_counts.values,
        names=file_type_counts.index,
        title="File Type Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)


def progress_bar_component(current_step: int, total_steps: int, step_name: str):
    """Display progress bar for processing steps"""
    progress = current_step / total_steps
    st.progress(progress)
    st.write(f"Step {current_step}/{total_steps}: {step_name}")


def analytics_dashboard_component(results_history: List[Dict[str, Any]]):
    """Display analytics dashboard with historical data"""
    st.subheader("📈 Analytics Dashboard")
    
    if not results_history:
        st.info("No historical data available. Process some documents first!")
        return
    
    # Processing success rate
    total_processed = len(results_history)
    successful_processed = sum(1 for r in results_history if r.get("success"))
    success_rate = (successful_processed / total_processed) * 100 if total_processed > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Processed", total_processed)
    with col2:
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col3:
        st.metric("Failed", total_processed - successful_processed)
    
    # Transaction trends
    st.subheader("Transaction Trends")
    
    # Aggregate transaction data
    all_transaction_types = []
    for result in results_history:
        if result.get("success") and result.get("transaction_pairs"):
            for pair in result["transaction_pairs"]:
                all_transaction_types.append(pair["transaction_type"])
    
    if all_transaction_types:
        # Count frequency
        tx_counts = pd.Series(all_transaction_types).value_counts()
        
        # Create bar chart
        fig = px.bar(
            x=tx_counts.index,
            y=tx_counts.values,
            title="Most Common Transaction Types",
            labels={"x": "Transaction Type", "y": "Frequency"}
        )
        fig.update_layout(xaxis_tickangle=-45)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transaction data available for trends analysis") 