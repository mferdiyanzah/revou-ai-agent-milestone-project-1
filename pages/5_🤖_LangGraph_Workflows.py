"""
LangGraph Workflows Page for TReA System

This page provides visualization and management of LangGraph agentic workflows,
showing agent interactions, processing paths, and workflow statistics.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional

# Import TReA modules
from src.config import settings
from src.services.api_client import TReAAPIClient
from src.processors.langgraph_processor import LangGraphEnhancedProcessor
from src.services.monitoring import MonitoringService


def main():
    """Main function for LangGraph workflows page"""
    
    st.set_page_config(
        page_title="LangGraph Workflows - TReA",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 LangGraph Agentic Workflows")
    st.markdown("**Advanced Multi-Agent Processing for Treasury Documents**")
    st.divider()
    
    # Initialize services
    api_client = TReAAPIClient()
    monitoring = MonitoringService()
    
    try:
        processor = LangGraphEnhancedProcessor(api_client, monitoring)
        langgraph_available = processor.langgraph_available
    except Exception as e:
        st.error(f"Failed to initialize LangGraph processor: {str(e)}")
        langgraph_available = False
        processor = None
    
    # Status overview
    show_langgraph_status(processor, langgraph_available)
    
    if not langgraph_available:
        show_setup_instructions()
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔄 Workflow Architecture", 
        "📊 Workflow Analytics", 
        "🧪 Test Workflows",
        "📚 Workflow History"
    ])
    
    with tab1:
        show_workflow_architecture()
    
    with tab2:
        show_workflow_analytics(processor, monitoring)
    
    with tab3:
        show_test_workflows(processor)
    
    with tab4:
        show_workflow_history(processor)


def show_langgraph_status(processor, langgraph_available: bool):
    """Show LangGraph service status and configuration"""
    
    st.subheader("🔍 Service Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if langgraph_available:
            st.success("✅ LangGraph Service")
            st.caption("Active and Ready")
        else:
            st.error("❌ LangGraph Service")
            st.caption("Not Available")
    
    with col2:
        openai_status = "✅ Connected" if settings.openai_api_key else "❌ Missing API Key"
        st.metric("OpenAI Integration", openai_status)
    
    with col3:
        if processor and hasattr(processor, 'monitoring'):
            st.success("✅ Monitoring")
            st.caption("Active")
        else:
            st.warning("⚠️ Monitoring")
            st.caption("Limited")
    
    with col4:
        if processor and hasattr(processor, 'guardrails'):
            st.success("✅ Guardrails")
            st.caption("Enabled")
        else:
            st.warning("⚠️ Guardrails")
            st.caption("Disabled")
    
    # Detailed status if available
    if processor and langgraph_available:
        with st.expander("📋 Detailed Service Information"):
            status_info = processor.get_langgraph_status()
            st.json(status_info)


def show_setup_instructions():
    """Show setup instructions when LangGraph is not available"""
    
    st.error("🚫 LangGraph Service Not Available")
    
    st.markdown("""
    ### 🛠️ Setup Required
    
    To use LangGraph agentic workflows, please ensure:
    
    1. **OpenAI API Key**: Required for LLM operations
       ```bash
       # Add to your .env file
       OPENAI_API_KEY=your_openai_api_key_here
       ```
    
    2. **LangGraph Dependencies**: Should be installed automatically
       ```bash
       pip install langgraph langchain-core langchain-openai
       ```
    
    3. **System Configuration**: Restart the application after adding environment variables
    
    ### 🎯 Benefits of LangGraph Workflows
    
    - **📄 Document Processor Agent**: Specialized PDF and text analysis
    - **📋 Journal Mapping Agent**: Expert journal entry creation
    - **✅ Validation Agent**: Automated accuracy checking
    - **🔄 Multi-Agent Orchestration**: Intelligent workflow coordination
    - **💾 State Management**: Persistent workflow tracking
    - **🛡️ Enhanced Guardrails**: Treasury-aware security
    """)


def show_workflow_architecture():
    """Show the LangGraph workflow architecture and agent flow"""
    
    st.subheader("🔄 Agentic Workflow Architecture")
    
    # Create workflow diagram
    fig = go.Figure()
    
    # Define agent positions
    agents = {
        "START": {"x": 0, "y": 2, "color": "#90EE90", "symbol": "circle"},
        "Document Processor": {"x": 2, "y": 3, "color": "#87CEEB", "symbol": "square"},
        "Journal Mapper": {"x": 4, "y": 2, "color": "#DDA0DD", "symbol": "diamond"},
        "Validator": {"x": 6, "y": 1, "color": "#F0E68C", "symbol": "triangle-up"},
        "Error Handler": {"x": 4, "y": 0, "color": "#FFB6C1", "symbol": "cross"},
        "END": {"x": 8, "y": 2, "color": "#98FB98", "symbol": "circle"}
    }
    
    # Add nodes
    for agent, props in agents.items():
        fig.add_trace(go.Scatter(
            x=[props["x"]], 
            y=[props["y"]],
            mode='markers+text',
            marker=dict(
                size=40,
                color=props["color"],
                symbol=props["symbol"],
                line=dict(width=2, color='black')
            ),
            text=agent,
            textposition="bottom center",
            name=agent,
            showlegend=False
        ))
    
    # Add edges (workflow connections)
    edges = [
        ("START", "Document Processor"),
        ("Document Processor", "Journal Mapper"),
        ("Document Processor", "Error Handler"),
        ("Journal Mapper", "Validator"),
        ("Journal Mapper", "Error Handler"),
        ("Validator", "END"),
        ("Validator", "Error Handler"),
        ("Error Handler", "END")
    ]
    
    for start, end in edges:
        start_pos = agents[start]
        end_pos = agents[end]
        
        fig.add_trace(go.Scatter(
            x=[start_pos["x"], end_pos["x"]],
            y=[start_pos["y"], end_pos["y"]],
            mode='lines',
            line=dict(width=2, color='gray'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add arrow annotation
        fig.add_annotation(
            x=end_pos["x"],
            y=end_pos["y"],
            ax=start_pos["x"],
            ay=start_pos["y"],
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='gray',
            showarrow=True
        )
    
    fig.update_layout(
        title="LangGraph Treasury Processing Workflow",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        showlegend=False,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Agent descriptions
    st.subheader("🤖 Agent Specifications")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📄 Document Processor Agent**
        - Parses treasury documents
        - Extracts transaction data
        - Identifies patterns and structures
        - Handles PDF, text, CSV formats
        - Outputs structured transaction pairs
        """)
    
    with col2:
        st.markdown("""
        **📋 Journal Mapping Agent**
        - Creates double-entry journal mappings
        - Follows treasury accounting rules
        - Generates invoice/payment pairs
        - Uses standard account names
        - Provides business logic explanations
        """)
    
    with col3:
        st.markdown("""
        **✅ Validation Agent**
        - Verifies double-entry compliance
        - Checks account name accuracy
        - Validates business logic
        - Provides quality scores
        - Suggests improvements
        """)


def show_workflow_analytics(processor, monitoring: MonitoringService):
    """Show analytics and statistics for workflow executions"""
    
    st.subheader("📊 Workflow Analytics")
    
    # Generate sample analytics data (in production, this would come from actual monitoring)
    if processor and hasattr(processor, 'get_workflow_history'):
        workflow_history = processor.get_workflow_history(limit=50)
    else:
        # Sample data for demonstration
        workflow_history = generate_sample_workflow_data()
    
    if not workflow_history:
        st.info("No workflow execution history available yet.")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(workflow_history)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_workflows = len(df)
        st.metric("Total Workflows", total_workflows)
    
    with col2:
        success_rate = len(df[df['status'] == 'completed']) / len(df) * 100 if len(df) > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        avg_duration = df.get('duration', [30]).mean() if 'duration' in df.columns else 30
        st.metric("Avg Duration", f"{avg_duration:.1f}s")
    
    with col4:
        document_workflows = len(df[df['type'] == 'document']) if 'type' in df.columns else 0
        st.metric("Document Workflows", document_workflows)
    
    # Workflow type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if 'type' in df.columns:
            type_counts = df['type'].value_counts()
            fig_pie = px.pie(
                values=type_counts.values, 
                names=type_counts.index,
                title="Workflow Types Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        if 'status' in df.columns:
            status_counts = df['status'].value_counts()
            fig_bar = px.bar(
                x=status_counts.index,
                y=status_counts.values,
                title="Workflow Status Distribution",
                color=status_counts.index
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Recent workflow executions
    st.subheader("📋 Recent Workflow Executions")
    
    # Display workflow history table
    if workflow_history:
        display_df = pd.DataFrame(workflow_history)
        
        # Format timestamp if available
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(
            display_df, 
            use_container_width=True,
            hide_index=True
        )


def show_test_workflows(processor):
    """Provide interface for testing LangGraph workflows"""
    
    st.subheader("🧪 Test Agentic Workflows")
    
    if not processor or not processor.langgraph_available:
        st.warning("LangGraph service not available for testing.")
        return
    
    # Test options
    test_type = st.selectbox(
        "Select Test Type:",
        ["Manual Transaction Input", "Sample Document Processing", "Agent Performance Test"]
    )
    
    if test_type == "Manual Transaction Input":
        test_manual_input(processor)
    elif test_type == "Sample Document Processing":
        test_document_processing(processor)
    elif test_type == "Agent Performance Test":
        test_agent_performance(processor)


def test_manual_input(processor):
    """Test manual input processing with LangGraph"""
    
    st.markdown("### ✍️ Test Manual Transaction Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Transaction Types:**")
        txn_types = st.text_area(
            "Transaction Types",
            value="BUY\nSELL\nDIVIDEND",
            height=100,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**Asset Classes:**")
        asset_classes = st.text_area(
            "Asset Classes", 
            value="BOND\nSTOCK",
            height=100,
            label_visibility="collapsed"
        )
    
    if st.button("🚀 Test LangGraph Processing", type="primary"):
        if txn_types.strip() and asset_classes.strip():
            with st.spinner("Processing with LangGraph agents..."):
                
                # Parse inputs
                txn_list = [t.strip().upper() for t in txn_types.split('\n') if t.strip()]
                asset_list = [a.strip().upper() for a in asset_classes.split('\n') if a.strip()]
                
                # Process with LangGraph
                try:
                    results = processor.process_manual_input_with_langgraph(
                        transaction_types=txn_list,
                        asset_classes=asset_list,
                        user_id="test_user"
                    )
                    
                    if results.get("success"):
                        st.success("✅ LangGraph processing completed successfully!")
                        
                        # Show results summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            pairs_count = len(results.get("transaction_pairs", []))
                            st.metric("Transaction Pairs", pairs_count)
                        
                        with col2:
                            workflow_id = results.get("workflow_id", "N/A")
                            st.metric("Workflow ID", workflow_id)
                        
                        with col3:
                            processing_method = results.get("processing_method", "unknown")
                            st.metric("Method", processing_method)
                        
                        # Show journal suggestions if available
                        if results.get("journal_suggestions", {}).get("success"):
                            with st.expander("📋 Generated Journal Mappings"):
                                suggestions = results["journal_suggestions"].get("suggestions", "")
                                st.text(suggestions)
                        
                        # Show validation results if available
                        if results.get("validation_results", {}).get("success"):
                            with st.expander("✅ Validation Results"):
                                validation = results["validation_results"]
                                score = validation.get("validation_score", 0)
                                feedback = validation.get("feedback", "")
                                
                                st.metric("Validation Score", f"{score}/100")
                                if feedback:
                                    st.text(feedback)
                    
                    else:
                        st.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"❌ Test failed: {str(e)}")
        else:
            st.warning("Please enter both transaction types and asset classes.")


def test_document_processing(processor):
    """Test document processing workflows"""
    
    st.markdown("### 📄 Test Document Processing")
    
    # Sample documents to test with
    sample_docs = {
        "Treasury Statement Sample": """
        TREASURY STATEMENT
        Date: 2024-01-15
        
        TRANSACTIONS:
        BUY BOND USD 1,000,000
        SELL STOCK USD 500,000
        DIVIDEND INCOME USD 25,000
        INTEREST PAYMENT USD 15,000
        """,
        "Trade Confirmation": """
        TRADE CONFIRMATION
        
        Transaction: PURCHASE
        Security: GOVERNMENT BOND
        Amount: USD 2,000,000
        Settlement Date: 2024-01-20
        """,
        "Cash Flow Report": """
        CASH FLOW REPORT
        
        INFLOWS:
        - Redemption proceeds: USD 1,500,000
        - Coupon payments: USD 75,000
        
        OUTFLOWS:
        - New investments: USD 2,000,000
        - Management fees: USD 10,000
        """
    }
    
    selected_doc = st.selectbox("Select Sample Document:", list(sample_docs.keys()))
    
    # Show the sample document
    with st.expander("📄 Document Content"):
        st.text(sample_docs[selected_doc])
    
    if st.button("🚀 Process Sample Document", type="primary"):
        with st.spinner("Processing document with LangGraph workflow..."):
            try:
                # Save sample document as temporary file
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(sample_docs[selected_doc])
                    temp_file_path = f.name
                
                # Process with LangGraph
                results = processor.process_document(temp_file_path, use_langgraph=True)
                
                # Clean up temporary file
                os.unlink(temp_file_path)
                
                if results.get("success"):
                    st.success("✅ Document processing completed!")
                    
                    # Show workflow info
                    if results.get("agent_workflow"):
                        workflow = results["agent_workflow"]
                        st.info(f"🔄 Workflow ID: {workflow.get('workflow_id', 'N/A')}")
                        
                        processing_path = workflow.get("processing_path", [])
                        if processing_path:
                            path_display = " → ".join([step.replace('_', ' ').title() for step in processing_path])
                            st.caption(f"**Agent Path:** {path_display}")
                    
                    # Show extracted transactions
                    transaction_pairs = results.get("transaction_pairs", [])
                    if transaction_pairs:
                        st.subheader(f"📋 Extracted Transactions ({len(transaction_pairs)})")
                        df = pd.DataFrame(transaction_pairs)
                        st.dataframe(df, use_container_width=True)
                    
                    # Show journal suggestions
                    if results.get("journal_suggestions", {}).get("success"):
                        with st.expander("📋 Journal Mapping Suggestions"):
                            suggestions = results["journal_suggestions"].get("suggestions", "")
                            st.text(suggestions)
                
                else:
                    st.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"❌ Test failed: {str(e)}")


def test_agent_performance(processor):
    """Test individual agent performance"""
    
    st.markdown("### 🎯 Agent Performance Testing")
    
    st.info("This feature tests individual agents within the LangGraph workflow.")
    
    # Agent selection
    agent_to_test = st.selectbox(
        "Select Agent to Test:",
        ["Document Processor", "Journal Mapper", "Validator", "Full Workflow"]
    )
    
    # Test parameters
    col1, col2 = st.columns(2)
    
    with col1:
        test_iterations = st.number_input("Test Iterations:", min_value=1, max_value=10, value=3)
    
    with col2:
        timeout_seconds = st.number_input("Timeout (seconds):", min_value=5, max_value=60, value=30)
    
    if st.button("🧪 Run Performance Test", type="primary"):
        st.info(f"Performance testing for {agent_to_test} is not yet implemented in this demo.")
        st.markdown("""
        **Planned Performance Metrics:**
        - Response time per agent
        - Success/failure rates
        - Resource utilization
        - Output quality scores
        - Error frequency analysis
        """)


def show_workflow_history(processor):
    """Show detailed workflow execution history"""
    
    st.subheader("📚 Workflow Execution History")
    
    if not processor or not processor.langgraph_available:
        st.warning("LangGraph service not available for history retrieval.")
        return
    
    # Get workflow history
    try:
        history = processor.get_workflow_history(limit=20)
        
        if not history:
            st.info("No workflow execution history available.")
            return
        
        # Display history
        df = pd.DataFrame(history)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'type' in df.columns:
                type_filter = st.multiselect(
                    "Filter by Type:",
                    options=df['type'].unique(),
                    default=df['type'].unique()
                )
                df = df[df['type'].isin(type_filter)]
        
        with col2:
            if 'status' in df.columns:
                status_filter = st.multiselect(
                    "Filter by Status:",
                    options=df['status'].unique(),
                    default=df['status'].unique()
                )
                df = df[df['status'].isin(status_filter)]
        
        with col3:
            if 'timestamp' in df.columns:
                # Date range filter would go here
                st.caption("Date filtering available in full version")
        
        # Display filtered results
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export option
        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download History (CSV)",
                data=csv,
                file_name=f"workflow_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"Failed to retrieve workflow history: {str(e)}")


def generate_sample_workflow_data() -> List[Dict[str, Any]]:
    """Generate sample workflow data for demonstration"""
    
    import random
    
    workflows = []
    statuses = ['completed', 'failed', 'running']
    types = ['document', 'manual']
    agents = [
        ['document_processor', 'journal_mapping', 'validation'],
        ['journal_mapping', 'validation'],
        ['document_processor', 'error_handler'],
        ['journal_mapping', 'error_handler']
    ]
    
    for i in range(20):
        workflow = {
            'workflow_id': f'workflow_{i+1}',
            'type': random.choice(types),
            'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            'status': random.choice(statuses),
            'agents_used': random.choice(agents),
            'duration': random.uniform(15, 120)
        }
        workflows.append(workflow)
    
    return workflows


if __name__ == "__main__":
    main() 