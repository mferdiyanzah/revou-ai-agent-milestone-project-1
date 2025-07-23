"""
TReA - Treasury with Embedded AI
Main Application Entry Point with LangGraph Integration

This is the main entry point for the TReA AI application.
Single-page flow: Upload PDF or Input Transaction Data → AI Analysis → Immediate Results
Now with LangGraph agentic workflows for enhanced multi-agent processing.

Run with: python main.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import traceback
import sys
import os

# Import our modules
from src.config import settings
from src.services.api_client import TReAAPIClient
from src.processors.ai_processor import AIEnhancedProcessor
from src.processors.langgraph_processor import LangGraphEnhancedProcessor

# Initialize session state
if "results_history" not in st.session_state:
    st.session_state.results_history = []
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "use_langgraph" not in st.session_state:
    st.session_state.use_langgraph = settings.auto_enable_langgraph  # Auto-enable based on config

def main():
    """Main application function with single-page flow and LangGraph integration"""
    
    # Configure page
    st.set_page_config(
        page_title="TReA - Treasury AI",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🏦 TReA - Treasury with Embedded AI")
    st.markdown("**AI-Powered Treasury Document Processing with LangGraph Agentic Workflows**")
    st.divider()
    
    # Enhanced sidebar with LangGraph options
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Configuration
        api_base_url = st.text_input("API Base URL", value=settings.api_base_url)
        api_token = st.text_input("API Token", type="password", value=settings.api_token or "")
        
        # LangGraph Configuration
        st.subheader("🤖 AI Processing Options")
        use_langgraph = st.checkbox(
            "Enable LangGraph Agentic Workflows", 
            value=st.session_state.use_langgraph,
            help="Use advanced multi-agent workflows for enhanced processing (Auto-enabled by default)"
        )
        st.session_state.use_langgraph = use_langgraph
        
        if use_langgraph:
            st.success("🌟 **Agentic Mode Active**\n\nUsing specialized AI agents:\n- 📄 Document Processor\n- 📋 Journal Mapper\n- ✅ Validation Agent")
        else:
            st.warning("📊 **Standard Mode**\n\nUsing traditional AI processing")
        
        # LangSmith Status
        st.subheader("🔍 LangSmith Tracing")
        if settings.langsmith_configured:
            st.success(f"✅ **Active**\n\nProject: `{settings.langchain_project}`")
            st.caption("Workflow tracing enabled for enhanced monitoring")
        else:
            st.info("ℹ️ **Not Configured**\n\nAdd `LANGCHAIN_API_KEY` to .env for workflow tracing")
            if st.button("📖 LangSmith Setup Guide"):
                st.markdown("""
                **🔍 LangSmith Setup (Optional)**
                
                LangSmith provides advanced tracing and monitoring for LangGraph workflows:
                
                1. **Sign up**: Visit [smith.langchain.com](https://smith.langchain.com/)
                2. **Get API Key**: Create a new API key in your settings
                3. **Add to .env**:
                   ```
                   LANGCHAIN_TRACING_V2=true
                   LANGCHAIN_API_KEY=your_langsmith_api_key_here
                   LANGCHAIN_PROJECT=trea-treasury-ai
                   ```
                4. **Restart** the application
                
                **Benefits:**
                - 📊 Detailed workflow tracing
                - 🕐 Agent execution timelines  
                - 🐛 Advanced debugging
                - 📈 Performance analytics
                """)
        
        # Store in session state
        st.session_state.api_base_url = api_base_url
        st.session_state.api_token = api_token
    
    # Initialize API client and processor
    api_client = TReAAPIClient(
        base_url=st.session_state.get("api_base_url", settings.api_base_url),
        token=st.session_state.get("api_token", settings.api_token)
    )
    
    # Choose processor based on LangGraph setting
    if st.session_state.use_langgraph:
        processor = LangGraphEnhancedProcessor(api_client)
        processor_type = "LangGraph Enhanced"
    else:
        processor = AIEnhancedProcessor(api_client)
        processor_type = "Standard AI"
    
    # Show processor status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Processor Type", processor_type)
    
    with col2:
        if hasattr(processor, 'langgraph_available'):
            langgraph_status = "✅ Available" if processor.langgraph_available else "❌ Unavailable"
        else:
            langgraph_status = "N/A"
        st.metric("LangGraph Status", langgraph_status)
    
    with col3:
        openai_status = "✅ Available" if processor.openai_available else "❌ Unavailable"
        st.metric("OpenAI Status", openai_status)
    
    with col4:
        langsmith_status = "✅ Active" if settings.langsmith_configured else "ℹ️ Optional"
        st.metric("LangSmith Tracing", langsmith_status)
    
    # Main input section
    st.header("📥 Input Options")
    st.markdown("Choose how you want to provide transaction data for AI analysis:")
    
    # Create two tabs for input methods
    tab1, tab2 = st.tabs(["📄 Upload PDF Document", "✍️ Manual Transaction Input"])
    
    with tab1:
        st.markdown("### Upload Treasury Document")
        st.markdown("Upload a PDF treasury statement for automatic processing and analysis.")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a treasury statement PDF file"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{file_size_mb:.2f} MB")
            
            if st.button("🚀 Process PDF Document", type="primary", key="process_pdf"):
                process_pdf_document(uploaded_file, processor)
    
    with tab2:
        st.markdown("### Manual Transaction Input")
        st.markdown("Enter transaction types and asset classes manually for AI journal mapping suggestions.")
        
        with st.form("manual_input_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Transaction Types** (one per line):")
                transaction_types = st.text_area(
                    "Transaction Types",
                    placeholder="BUY\nSELL\nDIVIDEND\nINTEREST\nSUBSCRIPTION\nREDEMPTION",
                    height=150,
                    label_visibility="collapsed"
                )
            
            with col2:
                st.markdown("**Asset Classes** (one per line):")
                asset_classes = st.text_area(
                    "Asset Classes",
                    placeholder="STOCK\nBOND\nFUND\nETF\nCASH\nDERIVATIVE",
                    height=150,
                    label_visibility="collapsed"
                )
            
            analyze_manual = st.form_submit_button("🤖 Analyze with AI", type="primary")
        
        if analyze_manual:
            if not transaction_types.strip() or not asset_classes.strip():
                st.warning("⚠️ Please enter both transaction types and asset classes")
            else:
                process_manual_input(transaction_types, asset_classes, processor)


def process_pdf_document(uploaded_file, processor):
    """Process uploaded PDF document and show immediate results"""
    
    # Results container
    results_container = st.container()
    
    with results_container:
        # Show processing status
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        try:
            with status_placeholder:
                if hasattr(processor, 'langgraph_available') and processor.langgraph_available:
                    st.info("🤖 Processing PDF document with LangGraph agentic workflows...")
                else:
                    st.info("🔄 Processing PDF document with AI...")
            
            with progress_placeholder:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Processing steps (updated for LangGraph)
                if hasattr(processor, 'langgraph_available') and processor.langgraph_available:
                    steps = [
                        "Saving uploaded file",
                        "Extracting document content", 
                        "Document Processor Agent analyzing content",
                        "Journal Mapping Agent creating entries",
                        "Validation Agent checking accuracy"
                    ]
                else:
                    steps = [
                        "Saving uploaded file",
                        "Uploading to TReA backend",
                        "Extracting PDF content", 
                        "AI analysis and enhancement",
                        "Generating journal mappings"
                    ]
                
                for i, step in enumerate(steps):
                    progress_bar.progress((i + 1) / len(steps))
                    status_text.text(f"Step {i+1}/{len(steps)}: {step}")
                    
                    if i == 0:
                        # Save file
                        file_path = processor.save_uploaded_file(uploaded_file)
                    elif i == len(steps) - 1:
                        # Process document
                        if hasattr(processor, 'langgraph_available') and processor.langgraph_available:
                            results = processor.process_document(file_path, use_langgraph=True)
                        else:
                            results = processor.process_document(file_path)
            
            # Clear progress indicators
            status_placeholder.empty()
            progress_placeholder.empty()
            
            # Show results immediately
            display_results(results, "PDF Document", uploaded_file.name)
            
        except Exception as e:
            status_placeholder.empty()
            progress_placeholder.empty()
            st.error(f"❌ Processing failed: {str(e)}")
            if settings.debug:
                st.text("Debug Information:")
                st.text(traceback.format_exc())


def process_manual_input(transaction_types: str, asset_classes: str, processor):
    """Process manual input and show immediate AI analysis results"""
    
    # Results container
    results_container = st.container()
    
    with results_container:
        # Show processing status
        status_placeholder = st.empty()
        
        try:
            # Parse inputs
            txn_types = [t.strip().upper() for t in transaction_types.split('\n') if t.strip()]
            asset_cls = [a.strip().upper() for a in asset_classes.split('\n') if a.strip()]
            
            with status_placeholder:
                if hasattr(processor, 'langgraph_available') and processor.langgraph_available:
                    st.info("🤖 Processing with LangGraph Journal Mapping Agent...")
                else:
                    st.info("🤖 Analyzing transaction data with AI...")
            
            # Process with appropriate method
            if hasattr(processor, 'process_manual_input_with_langgraph') and processor.langgraph_available:
                results = processor.process_manual_input_with_langgraph(
                    transaction_types=txn_types,
                    asset_classes=asset_cls,
                    user_id="streamlit_user"
                )
            else:
                # Fallback to creating transaction pairs and using standard processing
                transaction_pairs = []
                for txn_type in txn_types:
                    for asset_class in asset_cls:
                        transaction_pairs.append({
                            "transaction_type": txn_type,
                            "asset_class": asset_class,
                            "category": "Manual Input",
                            "description": f"{txn_type} - {asset_class}"
                        })
                
                # Generate AI analysis
                with st.spinner("🤖 Generating AI journal mapping suggestions..."):
                    if not processor.openai_available:
                        st.error("🚫 OpenAI service is not configured. Please add OPENAI_API_KEY to your environment.")
                        return
                    
                    # Call with guardrails disabled for manual input to avoid treasury term false positives
                    journal_suggestions = processor.openai_service.suggest_journal_mappings(
                        transaction_pairs, 
                        enable_guardrails=False  # Disable guardrails for manual input to avoid treasury term false positives
                    )
                
                # Create results structure for manual input
                results = {
                    "success": True,
                    "input_type": "manual",
                    "processing_method": "standard_ai",
                    "transaction_pairs": transaction_pairs,
                    "journal_suggestions": journal_suggestions,
                    "manual_input": {
                        "transaction_types": txn_types,
                        "asset_classes": asset_cls,
                        "total_combinations": len(transaction_pairs)
                    }
                }
            
            # Clear status
            status_placeholder.empty()
            
            # Show results immediately
            display_results(results, "Manual Input", f"{len(txn_types)} transaction types × {len(asset_cls)} asset classes")
            
        except Exception as e:
            status_placeholder.empty()
            st.error(f"❌ Analysis failed: {str(e)}")
            if settings.debug:
                st.text("Debug Information:")
                st.text(traceback.format_exc())


def display_results(results, input_method: str, source_info: str):
    """Display AI analysis results immediately with LangGraph workflow information"""
    
    if not results.get("success"):
        if results.get("error"):
            st.error(f"❌ {results['error']}")
        return
    
    # Store results in history
    st.session_state.current_results = results
    st.session_state.results_history.append({
        "timestamp": datetime.now(),
        "input_method": input_method,
        "source_info": source_info,
        "results": results
    })
    
    # Results header with processing method info
    processing_method = results.get("processing_method", "unknown")
    if processing_method == "langgraph_agentic":
        st.success(f"✅ AI Analysis Complete - {input_method} (🤖 LangGraph Agentic)")
    else:
        st.success(f"✅ AI Analysis Complete - {input_method} (📊 Standard AI)")
    
    # Show workflow information if available
    if results.get("agent_workflow"):
        workflow_info = results["agent_workflow"]
        st.info(f"🔄 **Workflow ID:** {workflow_info.get('workflow_id', 'N/A')}")
        
        # Show processing path
        processing_path = workflow_info.get("processing_path", [])
        if processing_path:
            path_emojis = {
                "document_processor": "📄",
                "journal_mapping": "📋", 
                "validation": "✅",
                "error_handler": "❌"
            }
            path_display = " → ".join([f"{path_emojis.get(step, '🔹')} {step.replace('_', ' ').title()}" for step in processing_path])
            st.caption(f"**Agent Path:** {path_display}")
    
    st.divider()
    
    # Summary metrics
    display_summary_metrics(results)
    
    # AI Journal Suggestions (Most Important)
    display_journal_suggestions(results)
    
    # Validation results if available
    if results.get("validation_results"):
        display_validation_results(results)
    
    # Transaction details (if available)
    if results.get("transaction_pairs"):
        display_transaction_details(results)
    
    # Additional AI insights (if available)
    display_ai_insights(results)


def display_validation_results(results):
    """Display validation results from LangGraph validation agent"""
    
    st.subheader("✅ Validation Results")
    
    validation_results = results.get("validation_results", {})
    
    if validation_results.get("success"):
        validation_score = validation_results.get("validation_score", 0)
        
        # Show validation score with color coding
        if validation_score >= 90:
            st.success(f"🌟 Validation Score: {validation_score}/100 - Excellent")
        elif validation_score >= 80:
            st.success(f"✅ Validation Score: {validation_score}/100 - Good")
        elif validation_score >= 70:
            st.warning(f"⚠️ Validation Score: {validation_score}/100 - Needs Review")
        else:
            st.error(f"❌ Validation Score: {validation_score}/100 - Requires Attention")
        
        # Show feedback
        feedback = validation_results.get("feedback", "")
        if feedback:
            with st.expander("📝 Detailed Validation Feedback"):
                st.write(feedback)
        
        # Show recommendations
        recommendations = validation_results.get("recommendations", [])
        if recommendations:
            with st.expander("💡 Recommendations"):
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
    
    elif validation_results.get("error"):
        st.warning(f"⚠️ Validation failed: {validation_results['error']}")
    else:
        st.info("ℹ️ No validation results available")
    
    st.divider()


def display_summary_metrics(results):
    """Display summary metrics with enhanced LangGraph information"""
    
    st.subheader("📊 Analysis Summary")
    
    transaction_pairs = results.get("transaction_pairs", [])
    unique_types = len(set(pair["transaction_type"] for pair in transaction_pairs))
    unique_classes = len(set(pair["asset_class"] for pair in transaction_pairs))
    processing_method = results.get("processing_method", "unknown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pairs", len(transaction_pairs))
    with col2:
        st.metric("Transaction Types", unique_types)
    with col3:
        st.metric("Asset Classes", unique_classes)
    with col4:
        method_display = {
            "langgraph_agentic": "🤖 Agentic",
            "standard_ai": "📊 Standard",
            "fallback_manual": "⚙️ Fallback"
        }.get(processing_method, processing_method)
        st.metric("Processing Method", method_display)


def display_journal_suggestions(results):
    """Display AI-generated journal mapping suggestions"""
    
    st.subheader("💡 AI Journal Mapping Suggestions")
    
    journal_suggestions = results.get("journal_suggestions", {})
    
    if journal_suggestions.get("success"):
        # Show model info
        model_used = journal_suggestions.get("model_used", "AI Model")
        unique_pairs = journal_suggestions.get("unique_pairs_count", 0)
        agent_info = journal_suggestions.get("agent", "")
        
        # Enhanced status message for LangGraph
        if agent_info == "journal_mapping_agent":
            st.success(f"🤖 Generated suggestions by Journal Mapping Agent for {unique_pairs} unique transaction types using {model_used}")
        elif results.get("processing_method") == "langgraph_agentic":
            st.success(f"🤖 Generated AI suggestions for {unique_pairs} unique transaction types using {model_used} (LangGraph Enhanced)")
        elif results.get("input_type") == "manual":
            st.success(f"✍️ Generated AI suggestions for {unique_pairs} unique transaction types using {model_used} (guardrails disabled for treasury term compatibility)")
        else:
            st.success(f"🤖 Generated AI suggestions for {unique_pairs} unique transaction types using {model_used}")
        
        # Display suggestions
        suggestions_text = journal_suggestions.get("suggestions", "")
        if suggestions_text:
            with st.container():
                st.markdown("### 📋 Recommended Journal Entries")
                st.markdown(suggestions_text)
                
                # Add copy button for suggestions
                if st.button("📋 Copy Suggestions to Clipboard", key="copy_suggestions"):
                    st.code(suggestions_text, language="text")
                    st.success("✅ Suggestions copied to clipboard area above")
        
        # Show any guardrail warnings if present
        if journal_suggestions.get("sanitized"):
            st.warning("🛡️ Content was automatically sanitized by security guardrails")
            if journal_suggestions.get("guardrail_warning"):
                with st.expander("View Guardrail Details"):
                    st.write(f"**Warning:** {journal_suggestions['guardrail_warning']}")
                    st.info("💡 This is normal for treasury content containing technical terms like account codes or references.")
    
    elif journal_suggestions.get("error"):
        st.error(f"❌ Journal suggestions failed: {journal_suggestions['error']}")
    else:
        st.warning("⚠️ No journal suggestions generated")
    
    st.divider()


def display_transaction_details(results):
    """Display transaction details in a table"""
    
    st.subheader("📋 Transaction Details")
    
    transaction_pairs = results.get("transaction_pairs", [])
    
    if transaction_pairs:
        # Create DataFrame
        df = pd.DataFrame(transaction_pairs)
        
        # Add processing source if available
        if "source" in df.columns:
            source_counts = df["source"].value_counts()
            st.caption(f"**Sources:** {', '.join([f'{source}: {count}' for source, count in source_counts.items()])}")
        
        # Add filters
        col1, col2 = st.columns(2)
        
        with col1:
            unique_types = df["transaction_type"].unique()
            selected_types = st.multiselect(
                "Filter by Transaction Type:",
                options=unique_types,
                default=unique_types,
                key="type_filter"
            )
        
        with col2:
            unique_classes = df["asset_class"].unique()
            selected_classes = st.multiselect(
                "Filter by Asset Class:",
                options=unique_classes,
                default=unique_classes,
                key="class_filter"
            )
        
        # Apply filters
        filtered_df = df[
            (df["transaction_type"].isin(selected_types)) &
            (df["asset_class"].isin(selected_classes))
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
            label="📥 Download Transaction Data (CSV)",
            data=csv,
            file_name=f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    st.divider()


def display_ai_insights(results):
    """Display additional AI insights"""
    
    # AI Analysis
    if results.get("ai_analysis", {}).get("success"):
        st.subheader("🧠 AI Analysis Insights")
        with st.expander("View Detailed AI Analysis"):
            st.write(results["ai_analysis"]["analysis"])
    
    # Similar Transactions (if available from PDF processing)
    if results.get("similar_transactions"):
        st.subheader("🔍 Similar Transactions Found")
        with st.expander("View Similar Transactions"):
            for item in results["similar_transactions"]:
                st.write(f"**Transaction:** {item['transaction']['description']}")
                st.write("**Similar:**")
                for similar in item['similar']:
                    st.write(f"- {similar['description']} (Similarity: {similar['similarity']:.2f})")
                st.divider()
    
    # Transaction Definitions
    if results.get("transaction_definitions", {}).get("success"):
        definitions_found = results["transaction_definitions"].get("definitions_found", 0)
        if definitions_found > 0:
            st.subheader("📖 Transaction Definitions")
            st.success(f"Found definitions for {definitions_found} transaction types")
            
            with st.expander("View All Definitions"):
                for key, definition_data in results["transaction_definitions"]["transactions"].items():
                    if definition_data.get("success") and definition_data.get("definitions"):
                        transaction_type = definition_data["transaction_type"]
                        asset_class = definition_data["asset_class"]
                        
                        st.write(f"**{transaction_type} ({asset_class})**")
                        st.write(definition_data["summary"])
                        
                        if definition_data["definitions"]:
                            st.write("**Sources:**")
                            for i, defn in enumerate(definition_data["definitions"][:2]):
                                st.write(f"{i+1}. [{defn['title']}]({defn['url']})")
                                st.write(f"   {defn['description'][:200]}...")
                        st.divider()


# Run the main function when the script is executed
if __name__ == "__main__":
    # Check if running with Streamlit
    if "streamlit" in sys.modules or os.environ.get("STREAMLIT_SERVER_HEADLESS"):
        # Running with Streamlit - execute main function
        main()
    else:
        # Running directly with Python - show instructions and launch Streamlit
        print("🏦 TReA - Treasury with Embedded AI")
        print("=" * 50)
        print("🎉 AI-Powered Treasury Document Processing with LangGraph")
        print("")
        print("Features:")
        print("  📄 PDF Processing - Upload treasury statements")
        print("  ✍️ Manual Input - Enter transaction types manually")
        print("  🤖 LangGraph Workflows - Multi-agent processing")
        print("  📋 Journal Mapping - Automated journal suggestions")
        print("  ✅ Validation - Agent-based result verification")
        print("")
        print("Starting Streamlit application...")
        print("=" * 50)
        
        # Launch Streamlit
        import subprocess
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
