"""
Vector Embeddings Management Page
Manages transaction type-asset class pairs and their vector embeddings for TReA system
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import traceback
import time
from io import StringIO

# Import our modules
from src.config import settings
from src.services.api_client import TReAAPIClient
from src.services.openai_service import OpenAIService
from src.services.vector_db import VectorDBService
from src.processors.ai_processor import AIEnhancedProcessor
from src.processors.brave_processor import BraveSearchProcessor, search_multiple_transactions_sync

# Page configuration
st.set_page_config(
    page_title="Vector Embeddings - TReA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Vector Embeddings Management")
st.markdown("Upload and manage transaction type-asset class pairs with their embeddings and definitions")

# Initialize services
@st.cache_resource
def initialize_services():
    """Initialize all services with caching"""
    api_client = TReAAPIClient()
    
    try:
        openai_service = OpenAIService()
        openai_available = True
    except ValueError:
        openai_service = None
        openai_available = False
    
    try:
        vector_db = VectorDBService()
        vector_db_available = True
    except Exception:
        vector_db = None
        vector_db_available = False
    
    try:
        brave_processor = BraveSearchProcessor(settings.brave_api_key) if settings.brave_search_enabled else None
        brave_available = bool(brave_processor)
    except Exception:
        brave_processor = None
        brave_available = False
    
    ai_processor = AIEnhancedProcessor(api_client, openai_service, vector_db, brave_processor)
    
    return {
        "api_client": api_client,
        "openai_service": openai_service,
        "vector_db": vector_db,
        "brave_processor": brave_processor,
        "ai_processor": ai_processor,
        "openai_available": openai_available,
        "vector_db_available": vector_db_available,
        "brave_available": brave_available
    }

services = initialize_services()

# Initialize session state
if "transaction_pairs" not in st.session_state:
    st.session_state.transaction_pairs = []
if "embeddings_data" not in st.session_state:
    st.session_state.embeddings_data = {}
if "definitions_data" not in st.session_state:
    st.session_state.definitions_data = {}

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Service status
    st.subheader("Service Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if services["openai_available"]:
            st.success("✅ OpenAI")
        else:
            st.error("❌ OpenAI")
    
    with col2:
        if services["vector_db_available"]:
            st.success("✅ Vector DB")
        else:
            st.error("❌ Vector DB")
    
    if services["brave_available"]:
        st.success("✅ Brave Search")
    else:
        st.error("❌ Brave Search")
    
    st.divider()
    
    # Configuration options
    st.subheader("Embedding Options")
    
    # Display current embedding model
    if services["openai_available"]:
        st.info(f"📊 Embedding Model: {settings.openai_embedding_model}")
    
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05
    )
    
    batch_size = st.number_input(
        "Batch Size",
        min_value=1,
        max_value=100,
        value=10
    )

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload CSV", "🧮 Generate Embeddings", "🔍 Search & Browse", "📊 Analytics"])

# Tab 1: CSV Upload
with tab1:
    st.header("📤 Upload Transaction Pairs CSV")
    st.markdown("Upload a CSV file containing transaction type and asset class pairs")
    
    # CSV format help
    with st.expander("📋 CSV Format Requirements"):
        st.markdown("""
        **Required Columns:**
        - `transaction_type`: The type of transaction (e.g., "Buy", "Sell", "Redemption")
        - `asset_class_id`: The asset class ID (1=Cash, 2=Fixed Income, 3=Fund, 5=Equity, 6=Time Deposit)
        
        **Optional Columns:**
        - `transaction_subtype`: More specific transaction subtype
        - `transaction_subtype_code`: Code for the subtype (e.g., "BUY", "SELL", "RDM")
        - `description`: Additional description for the pair
        - `category`: Transaction category (Trading, Investment, Income, etc.)
        - `priority`: Priority level (1-5)
        
        **Asset Class IDs:**
        - **1**: Cash & Cash Equivalents
        - **2**: Fixed Income Securities  
        - **3**: Fund Investments
        - **5**: Equity Securities
        - **6**: Time Deposits
        
        **Example CSV:**
        ```
        transaction_type,asset_class_id,transaction_subtype,transaction_subtype_code,description,category
        Buy,2,Buy,BUY,Purchase of fixed income securities,Trading
        Redemption,3,Redemption,RDM,Fund redemption,Investment
        Interest,2,Interest,INT,Interest payment received,Income
        ```
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload a CSV file with transaction type and asset class pairs"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = ['transaction_type', 'asset_class_id']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            else:
                st.success("✅ CSV file loaded successfully!")
                
                # Display data preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Data statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Unique Transaction Types", df['transaction_type'].nunique())
                with col3:
                    st.metric("Unique Asset Classes", df['asset_class_id'].nunique())
                
                # Show processing success message if flag is set
                if st.session_state.get("processing_success", False):
                    st.success(f"✅ Successfully processed {st.session_state.get('last_processed_count', 0)} transaction pairs! Check the 'Generate Embeddings' tab to view them.")
                    # Clear the flag so it doesn't show again
                    st.session_state.processing_success = False
                
                # Debug: Test button functionality
                col1, col2 = st.columns(2)
                with col1:
                    # Process data
                    if st.button("🚀 Process CSV Data", type="primary"):
                        with st.spinner("Processing CSV data..."):
                            st.write("🔄 Starting processing...")
                            # Clean and prepare data
                            df_clean = df.dropna(subset=['transaction_type', 'asset_class_id'])
                            df_clean = df_clean.drop_duplicates(subset=['transaction_type', 'asset_class_id'])
                            st.write(f"📊 Cleaned data: {len(df_clean)} rows after removing duplicates")
                            
                            # Convert to transaction pairs format
                            transaction_pairs = []
                            
                            # Map asset class ID to name
                            asset_class_mapping = {
                                "1": "Cash & Cash Equivalents",
                                "2": "Fixed Income Securities", 
                                "3": "Fund Investments",
                                "5": "Equity Securities",
                                "6": "Time Deposits"
                            }
                            
                            try:
                                for index, row in df_clean.iterrows():
                                    asset_class_id = str(row['asset_class_id']).strip()
                                    asset_class_name = asset_class_mapping.get(asset_class_id, f"Asset Class {asset_class_id}")
                                    
                                    pair = {
                                        'transaction_type': str(row['transaction_type']).strip(),
                                        'asset_class_id': asset_class_id,
                                        'asset_class_name': asset_class_name,
                                        'transaction_subtype': str(row.get('transaction_subtype', '')).strip() if pd.notna(row.get('transaction_subtype')) else '',
                                        'transaction_subtype_code': str(row.get('transaction_subtype_code', '')).strip() if pd.notna(row.get('transaction_subtype_code')) else '',
                                        'description': str(row.get('description', '')).strip() if pd.notna(row.get('description')) else '',
                                        'category': str(row.get('category', 'General')).strip() if pd.notna(row.get('category')) else 'General',
                                        'priority': int(row.get('priority', 3)) if pd.notna(row.get('priority')) and str(row.get('priority', 3)).isdigit() else 3,
                                        'uploaded_at': datetime.now().isoformat()
                                    }
                                    
                                    # Create full description if not provided
                                    if not pair['description']:
                                        pair['description'] = f"{pair['transaction_type']} - {asset_class_name}"
                                        if pair['transaction_subtype']:
                                            pair['description'] += f" ({pair['transaction_subtype']})"
                                    
                                    transaction_pairs.append(pair)
                                    
                            except Exception as row_error:
                                st.error(f"Error processing row {index}: {str(row_error)}")
                                st.write(f"Row data: {dict(row)}")
                                pass  # Changed from continue to pass since we're not in a loop
                            
                            st.write(f"🔗 Created {len(transaction_pairs)} transaction pairs")
                            
                            # Store in session state
                            st.session_state.transaction_pairs = transaction_pairs
                            
                            # Set a processing success flag
                            st.session_state.processing_success = True
                            st.session_state.last_processed_count = len(transaction_pairs)
                            
                            st.success(f"✅ Processed {len(transaction_pairs)} unique transaction pairs!")
                            
                            # Show immediate feedback
                            st.write("💾 Data stored in session state")
                            st.write(f"🔄 Current session state has {len(st.session_state.transaction_pairs)} pairs")
                            
                                                    # Add a small delay before rerun
                        time.sleep(1)
                        st.rerun()
                
                with col2:
                    # Debug: Simple test button
                    if st.button("🧪 Test Button"):
                        st.success("Button clicked! The button functionality works.")
                        st.write(f"Current pairs in session: {len(st.session_state.get('transaction_pairs', []))}")
                        
                        # Add a test transaction pair
                        test_pair = {
                            'transaction_type': 'Test Buy',
                            'asset_class_id': '2',
                            'asset_class_name': 'Fixed Income Securities',
                            'transaction_subtype': 'Test',
                            'transaction_subtype_code': 'TEST',
                            'description': 'Test transaction pair',
                            'category': 'Testing',
                            'priority': 1,
                            'uploaded_at': datetime.now().isoformat()
                        }
                        
                        if 'transaction_pairs' not in st.session_state:
                            st.session_state.transaction_pairs = []
                        
                        st.session_state.transaction_pairs.append(test_pair)
                        st.success(f"Added test pair! Total pairs: {len(st.session_state.transaction_pairs)}")
                        st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.code(traceback.format_exc())
    
    # Manual entry option
    st.divider()
    st.subheader("✏️ Manual Entry")
    
    with st.form("manual_entry_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            manual_transaction_type = st.text_input("Transaction Type", placeholder="e.g., Buy")
            manual_transaction_subtype = st.text_input("Transaction Subtype", placeholder="e.g., Buy")
            manual_description = st.text_area("Description", placeholder="Optional description")
        
        with col2:
            manual_asset_class_id = st.selectbox("Asset Class", 
                options=["1", "2", "3", "5", "6"],
                format_func=lambda x: {
                    "1": "1 - Cash & Cash Equivalents",
                    "2": "2 - Fixed Income Securities", 
                    "3": "3 - Fund Investments",
                    "5": "5 - Equity Securities",
                    "6": "6 - Time Deposits"
                }[x]
            )
            manual_subtype_code = st.text_input("Subtype Code", placeholder="e.g., BUY")
            manual_category = st.selectbox("Category", ["Trading", "Investment", "Income", "Principal", "Cash Management", "Expense", "Payment", "Accrual"])
        
        if st.form_submit_button("➕ Add Pair"):
            if manual_transaction_type and manual_asset_class_id:
                # Map asset class ID to name
                asset_class_mapping = {
                    "1": "Cash & Cash Equivalents",
                    "2": "Fixed Income Securities", 
                    "3": "Fund Investments",
                    "5": "Equity Securities",
                    "6": "Time Deposits"
                }
                asset_class_name = asset_class_mapping[manual_asset_class_id]
                
                new_pair = {
                    'transaction_type': manual_transaction_type.strip(),
                    'asset_class_id': manual_asset_class_id,
                    'asset_class_name': asset_class_name,
                    'transaction_subtype': manual_transaction_subtype.strip() if manual_transaction_subtype else '',
                    'transaction_subtype_code': manual_subtype_code.strip() if manual_subtype_code else '',
                    'description': manual_description.strip() if manual_description else f"{manual_transaction_type} - {asset_class_name}",
                    'category': manual_category,
                    'priority': 3,
                    'uploaded_at': datetime.now().isoformat()
                }
                
                if new_pair not in st.session_state.transaction_pairs:
                    st.session_state.transaction_pairs.append(new_pair)
                    st.success("✅ Transaction pair added!")
                    st.rerun()
                else:
                    st.warning("⚠️ This pair already exists!")
            else:
                st.error("❌ Please fill in both transaction type and asset class ID")

# Tab 2: Generate Embeddings
with tab2:
    st.header("🧮 Generate Embeddings")
    
    if not st.session_state.transaction_pairs:
        st.info("📋 Please upload a CSV file or add transaction pairs manually in the Upload tab first.")
    else:
        # Display current pairs
        st.subheader("📋 Current Transaction Pairs")
        pairs_df = pd.DataFrame(st.session_state.transaction_pairs)
        st.dataframe(pairs_df, use_container_width=True, hide_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Pairs"):
                st.session_state.transaction_pairs = []
                st.session_state.embeddings_data = {}
                st.rerun()
        
        with col2:
            if st.button("📥 Download as CSV"):
                csv = pairs_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"transaction_pairs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        st.divider()
        
        # Generate embeddings section
        if services["openai_available"]:
            st.subheader("🚀 Generate Embeddings")
            
            # Options
            include_definitions = st.checkbox(
                "Include definitions from Brave Search",
                value=services["brave_available"],
                disabled=not services["brave_available"]
            )
            
            store_in_db = st.checkbox(
                "Store embeddings in vector database",
                value=services["vector_db_available"],
                disabled=not services["vector_db_available"]
            )
            
            if st.button("🧮 Generate All Embeddings", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    total_pairs = len(st.session_state.transaction_pairs)
                    
                    # Step 1: Get definitions if requested
                    definitions_results = {}
                    if include_definitions and services["brave_available"]:
                        status_text.text("🔍 Searching for definitions...")
                        definitions_results = search_multiple_transactions_sync(
                            st.session_state.transaction_pairs,
                            settings.brave_api_key
                        )
                        st.session_state.definitions_data = definitions_results
                        progress_bar.progress(0.3)
                    
                    # Step 2: Generate embeddings
                    status_text.text("🧮 Generating embeddings...")
                    embeddings_results = {}
                    
                    for i, pair in enumerate(st.session_state.transaction_pairs):
                        try:
                            # Create text for embedding
                            text_to_embed = pair['description']
                            
                            # Add definition if available
                            if include_definitions and definitions_results.get("success"):
                                pair_key = f"{pair['transaction_type']}_{pair['asset_class_id']}"
                                if pair_key in definitions_results.get("transactions", {}):
                                    definition_data = definitions_results["transactions"][pair_key]
                                    if definition_data.get("success") and definition_data.get("summary"):
                                        text_to_embed += f" Definition: {definition_data['summary']}"
                            
                            # Generate embedding
                            try:
                                embedding = services["openai_service"].create_embedding(text_to_embed)
                            except Exception as embedding_error:
                                st.error(f"❌ Error generating embedding for {pair['transaction_type']}: {str(embedding_error)}")
                                continue
                            
                            embeddings_results[f"{pair['transaction_type']}_{pair['asset_class_id']}"] = {
                                "pair": pair,
                                "text": text_to_embed,
                                "embedding": embedding,
                                "generated_at": datetime.now().isoformat()
                            }
                            
                            # Store in vector DB if requested
                            if store_in_db and services["vector_db_available"]:
                                services["vector_db"].store_transaction_embedding(
                                    transaction_id=f"{pair['transaction_type']}_{pair['asset_class_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                                    transaction_type=pair['transaction_type'],
                                    asset_class=pair['asset_class_id'],
                                    description=text_to_embed,
                                    embedding=embedding,
                                    metadata={
                                        "category": pair['category'],
                                        "priority": pair['priority'],
                                        "generated_at": datetime.now().isoformat()
                                    }
                                )
                            
                            # Update progress
                            progress = 0.3 + (0.7 * (i + 1) / total_pairs)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing {i+1}/{total_pairs}: {pair['transaction_type']} - {pair['asset_class_id']}")
                            
                        except Exception as e:
                            st.error(f"❌ Error processing {pair['transaction_type']} - {pair['asset_class_id']}: {str(e)}")
                    
                    # Store results
                    st.session_state.embeddings_data = embeddings_results
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Embeddings generation completed!")
                    
                    # Show summary
                    st.success(f"✅ Generated embeddings for {len(embeddings_results)} transaction pairs!")
                    
                    if include_definitions and definitions_results.get("definitions_found", 0) > 0:
                        st.success(f"📖 Found definitions for {definitions_results['definitions_found']} pairs")
                    
                    if store_in_db and services["vector_db_available"]:
                        st.success("💾 Embeddings stored in vector database")
                
                except Exception as e:
                    st.error(f"❌ Error generating embeddings: {str(e)}")
                    st.code(traceback.format_exc())
        
        else:
            st.warning("⚠️ OpenAI service is not available. Please configure OPENAI_API_KEY in your .env file.")

# Tab 3: Search & Browse
with tab3:
    st.header("🔍 Search & Browse Embeddings")
    
    if not st.session_state.embeddings_data:
        st.info("🧮 Please generate embeddings first in the Generate Embeddings tab.")
    else:
        # Search functionality
        st.subheader("🔍 Semantic Search")
        
        search_query = st.text_input(
            "Search query",
            placeholder="e.g., 'stock purchase transaction' or 'dividend payment'"
        )
        
        if search_query and services["openai_available"]:
            try:
                # Generate embedding for search query
                query_embedding = services["openai_service"].create_embedding(search_query)
                
                # Calculate similarities
                similarities = []
                for key, data in st.session_state.embeddings_data.items():
                    similarity = np.dot(query_embedding, data["embedding"])
                    similarities.append({
                        "key": key,
                        "similarity": similarity,
                        "pair": data["pair"],
                        "text": data["text"]
                    })
                
                # Sort by similarity
                similarities.sort(key=lambda x: x["similarity"], reverse=True)
                
                # Display results
                st.subheader("🎯 Search Results")
                for i, result in enumerate(similarities[:10]):
                    if result["similarity"] >= similarity_threshold:
                        with st.expander(f"#{i+1} - {result['pair']['transaction_type']} | {result['pair']['asset_class_id']} (Similarity: {result['similarity']:.3f})"):
                            st.write(f"**Description:** {result['pair']['description']}")
                            st.write(f"**Category:** {result['pair']['category']}")
                            st.write(f"**Full Text:** {result['text'][:500]}...")
                            
                            # Show definition if available
                            if st.session_state.definitions_data.get("success"):
                                pair_key = f"{result['pair']['transaction_type']}_{result['pair']['asset_class_id']}"
                                if pair_key in st.session_state.definitions_data.get("transactions", {}):
                                    definition_data = st.session_state.definitions_data["transactions"][pair_key]
                                    if definition_data.get("success") and definition_data.get("summary"):
                                        st.write(f"**Definition:** {definition_data['summary'][:300]}...")
            
            except Exception as e:
                st.error(f"❌ Search error: {str(e)}")
        
        st.divider()
        
        # Browse all embeddings
        st.subheader("📊 Browse All Embeddings")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            transaction_types = list(set(data["pair"]["transaction_type"] for data in st.session_state.embeddings_data.values()))
            selected_transaction_types = st.multiselect(
                "Transaction Types",
                transaction_types,
                default=transaction_types
            )
        
        with col2:
            asset_classes = list(set(data["pair"]["asset_class_id"] for data in st.session_state.embeddings_data.values()))
            selected_asset_classes = st.multiselect(
                "Asset Classes",
                asset_classes,
                default=asset_classes
            )
        
        with col3:
            categories = list(set(data["pair"]["category"] for data in st.session_state.embeddings_data.values()))
            selected_categories = st.multiselect(
                "Categories",
                categories,
                default=categories
            )
        
        # Filter and display
        filtered_data = []
        for key, data in st.session_state.embeddings_data.items():
            pair = data["pair"]
            if (pair["transaction_type"] in selected_transaction_types and
                pair["asset_class_id"] in selected_asset_classes and
                pair["category"] in selected_categories):
                filtered_data.append({
                    "Transaction Type": pair["transaction_type"],
                    "Asset Class": pair["asset_class_id"],
                    "Category": pair["category"],
                    "Description": pair["description"],
                    "Generated At": data["generated_at"]
                })
        
        if filtered_data:
            filtered_df = pd.DataFrame(filtered_data)
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            # Download filtered data
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name=f"filtered_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No data matches the selected filters.")

# Tab 4: Analytics
with tab4:
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.embeddings_data:
        st.info("🧮 Please generate embeddings first to view analytics.")
    else:
        # Basic statistics
        total_embeddings = len(st.session_state.embeddings_data)
        
        # Extract data for analysis
        transaction_types = [data["pair"]["transaction_type"] for data in st.session_state.embeddings_data.values()]
        asset_classes = [data["pair"]["asset_class_id"] for data in st.session_state.embeddings_data.values()]
        categories = [data["pair"]["category"] for data in st.session_state.embeddings_data.values()]
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Embeddings", total_embeddings)
        with col2:
            st.metric("Unique Transaction Types", len(set(transaction_types)))
        with col3:
            st.metric("Unique Asset Classes", len(set(asset_classes)))
        with col4:
            st.metric("Unique Categories", len(set(categories)))
        
        # Distribution charts
        import plotly.express as px
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction types distribution
            transaction_counts = pd.Series(transaction_types).value_counts()
            fig1 = px.pie(
                values=transaction_counts.values,
                names=transaction_counts.index,
                title="Transaction Types Distribution"
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Asset classes distribution
            asset_counts = pd.Series(asset_classes).value_counts()
            fig2 = px.pie(
                values=asset_counts.values,
                names=asset_counts.index,
                title="Asset Classes Distribution"
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Category distribution
        category_counts = pd.Series(categories).value_counts()
        fig3 = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Categories Distribution",
            labels={"x": "Category", "y": "Count"}
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Embedding quality metrics
        if services["openai_available"]:
            st.subheader("🎯 Embedding Quality Metrics")
            
            try:
                # Calculate pairwise similarities
                embeddings_matrix = np.array([data["embedding"] for data in st.session_state.embeddings_data.values()])
                similarity_matrix = np.dot(embeddings_matrix, embeddings_matrix.T)
                
                # Remove diagonal (self-similarity)
                np.fill_diagonal(similarity_matrix, 0)
                
                # Statistics
                avg_similarity = np.mean(similarity_matrix)
                max_similarity = np.max(similarity_matrix)
                min_similarity = np.min(similarity_matrix)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Similarity", f"{avg_similarity:.3f}")
                with col2:
                    st.metric("Max Similarity", f"{max_similarity:.3f}")
                with col3:
                    st.metric("Min Similarity", f"{min_similarity:.3f}")
                
                # Similarity distribution
                similarities_flat = similarity_matrix.flatten()
                similarities_flat = similarities_flat[similarities_flat != 0]  # Remove zeros
                
                fig4 = px.histogram(
                    x=similarities_flat,
                    title="Similarity Score Distribution",
                    labels={"x": "Similarity Score", "y": "Frequency"},
                    nbins=50
                )
                st.plotly_chart(fig4, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error calculating similarity metrics: {str(e)}")
        
        # System status
        st.divider()
        st.subheader("🔧 System Status")
        
        status_data = []
        for service_name, available in [
            ("OpenAI Service", services["openai_available"]),
            ("Vector Database", services["vector_db_available"]),
            ("Brave Search", services["brave_available"])
        ]:
            status_data.append({
                "Service": service_name,
                "Status": "✅ Available" if available else "❌ Not Available"
            })
        
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.markdown("💡 **Tips:** Use this page to build and manage your transaction pair embeddings for improved journal mapping accuracy.") 