"""
Definition Search Management Page
Search and manage transaction definitions using Brave Search for TReA system
"""

import streamlit as st
import pandas as pd
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List

# Import our modules
from src.config import settings
from src.processors.brave_processor import BraveSearchProcessor, search_transaction_definition_sync, search_multiple_transactions_sync
import os


# Journal Setup Suggestion Functions (moved to top to avoid NameError)
def load_asset_class_mapping():
    """Load asset class mapping from JSON file"""
    try:
        mapping_path = os.path.join("data", "asset_class_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    except Exception:
        return {}

def suggest_journal_setup(transaction_type: str, asset_class: str, definition_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest complete journal setup based on transaction type, asset class, and definition
    Returns multiple journal entries (Invoice + Payment/Refund) as seen in Master Journal.csv
    
    Args:
        transaction_type: Type of transaction (e.g., "Buy", "Sell", "Dividend")
        asset_class: Asset class ID or name
        definition_result: Search result containing definitions
    
    Returns:
        Complete journal setup with all related entries (Invoice, Payment, Debit Memo, etc.)
    """
    mapping = load_asset_class_mapping()
    
    # Determine transaction category based on transaction type
    transaction_lower = transaction_type.lower()
    journal_entries = []
    
    if transaction_lower in ['buy', 'purchase', 'subscription']:
        category = "Trading"
        
        # Step 1: Invoice (Initial Recognition)
        journal_entries.append({
            "journal_type": "Invoice",
            "subtype_code": "BUY",
            "description": f"{transaction_type} - Initial Recognition",
            "entries": [
                {
                    "account_class": "Marketable Securities",
                    "account_subclass": "Investment - Fixed Income" if asset_class == "2" else "Investment - Equity" if asset_class == "5" else "Investment - Funds",
                    "account_type": "DR",
                    "formula_expression": "AMOUNT",
                    "weight": 1
                },
                {
                    "account_class": "Interest Receivable", 
                    "account_subclass": "Interest Receivable - Bought",
                    "account_type": "DR",
                    "formula_expression": "ACCRUED_INTEREST",
                    "weight": 2
                },
                {
                    "account_class": "Other payable",
                    "account_subclass": "Other payable", 
                    "account_type": "CR",
                    "formula_expression": "TOTAL_AMOUNT",
                    "weight": 3
                }
            ]
        })
        
        # Step 2: Payment Created (Cash Settlement)
        journal_entries.append({
            "journal_type": "Payment Created",
            "subtype_code": "BUY_PC",
            "description": f"{transaction_type} - Cash Settlement",
            "entries": [
                {
                    "account_class": "Other payable",
                    "account_subclass": "Other payable",
                    "account_type": "DR", 
                    "formula_expression": "TOTAL_AMOUNT",
                    "weight": 1
                },
                {
                    "account_class": "Cash for Investment",
                    "account_subclass": "Cash for Investment",
                    "account_type": "CR",
                    "formula_expression": "TOTAL_AMOUNT", 
                    "weight": 2
                }
            ]
        })
        
    elif transaction_lower in ['sell', 'disposal']:
        category = "Trading"
        
        # Step 1: Debit Memo (Asset Recognition)
        journal_entries.append({
            "journal_type": "Debit Memo",
            "subtype_code": "SELL",
            "description": f"{transaction_type} - Asset Disposal",
            "entries": [
                {
                    "account_class": "Investment Receivable",
                    "account_subclass": "Investment Receivable",
                    "account_type": "DR",
                    "formula_expression": "GROSS_AMOUNT",
                    "weight": 1
                },
                {
                    "account_class": "Other income" if transaction_lower == "sell" else "Investment in Financial Instrument",
                    "account_subclass": "Other income" if transaction_lower == "sell" else "Investment - Capital Gain/Loss", 
                    "account_type": "CR",
                    "formula_expression": "GAIN_LOSS",
                    "weight": 2
                },
                {
                    "account_class": "Marketable Securities",
                    "account_subclass": "Investment - Fixed Income" if asset_class == "2" else "Investment - Equity" if asset_class == "5" else "Investment - Funds",
                    "account_type": "CR",
                    "formula_expression": "BOOK_VALUE",
                    "weight": 3
                }
            ]
        })
        
        # Step 2: Refund Created (Cash Receipt)
        journal_entries.append({
            "journal_type": "Refund Created", 
            "subtype_code": "SELL_RC",
            "description": f"{transaction_type} - Cash Receipt",
            "entries": [
                {
                    "account_class": "Cash for Investment",
                    "account_subclass": "Cash for Investment",
                    "account_type": "DR",
                    "formula_expression": "NET_PROCEEDS",
                    "weight": 1
                },
                {
                    "account_class": "Investment Receivable",
                    "account_subclass": "Investment Receivable", 
                    "account_type": "CR",
                    "formula_expression": "GROSS_AMOUNT",
                    "weight": 2
                }
            ]
        })
        
    elif transaction_lower in ['dividend', 'dividends']:
        category = "Income"
        
        # Step 1: Debit Memo (Income Recognition)  
        journal_entries.append({
            "journal_type": "Debit Memo",
            "subtype_code": "DVD",
            "description": f"{transaction_type} - Income Recognition",
            "entries": [
                {
                    "account_class": "Investment Receivable",
                    "account_subclass": "Investment Receivable",
                    "account_type": "DR",
                    "formula_expression": "DIVIDEND_AMOUNT",
                    "weight": 1
                },
                {
                    "account_class": "Investment in Financial Instrument",
                    "account_subclass": "Investment - Dividend",
                    "account_type": "CR", 
                    "formula_expression": "DIVIDEND_AMOUNT",
                    "weight": 2
                }
            ]
        })
        
        # Step 2: Refund Created (Cash Receipt)
        journal_entries.append({
            "journal_type": "Refund Created",
            "subtype_code": "DVD_RC", 
            "description": f"{transaction_type} - Cash Receipt",
            "entries": [
                {
                    "account_class": "Cash for Investment",
                    "account_subclass": "Cash for Investment",
                    "account_type": "DR",
                    "formula_expression": "NET_DIVIDEND",
                    "weight": 1
                },
                {
                    "account_class": "Investment Receivable",
                    "account_subclass": "Investment Receivable",
                    "account_type": "CR",
                    "formula_expression": "DIVIDEND_AMOUNT",
                    "weight": 2
                }
            ]
        })
        
    elif transaction_lower in ['interest', 'coupon', 'coupons']:
        category = "Income"
        
        # Step 1: Debit Memo (Interest Recognition)
        journal_entries.append({
            "journal_type": "Debit Memo",
            "subtype_code": "INT",
            "description": f"{transaction_type} - Interest Recognition", 
            "entries": [
                {
                    "account_class": "Interest Receivable",
                    "account_subclass": "Interest Receivable",
                    "account_type": "DR",
                    "formula_expression": "INTEREST_GROSS",
                    "weight": 1
                },
                {
                    "account_class": "art. 23 final",
                    "account_subclass": "art. 23 final",
                    "account_type": "DR",
                    "formula_expression": "TAX_AMOUNT", 
                    "weight": 2
                },
                {
                    "account_class": "Interest income",
                    "account_subclass": "Interest income",
                    "account_type": "CR",
                    "formula_expression": "INTEREST_GROSS",
                    "weight": 3
                }
            ]
        })
        
        # Step 2: Refund Created (Cash Receipt)
        journal_entries.append({
            "journal_type": "Refund Created",
            "subtype_code": "INT_RC",
            "description": f"{transaction_type} - Cash Receipt",
            "entries": [
                {
                    "account_class": "Cash for Investment", 
                    "account_subclass": "Cash for Investment",
                    "account_type": "DR",
                    "formula_expression": "INTEREST_NET",
                    "weight": 1
                },
                {
                    "account_class": "Interest Receivable",
                    "account_subclass": "Interest Receivable",
                    "account_type": "CR",
                    "formula_expression": "INTEREST_GROSS",
                    "weight": 2
                }
            ]
        })
        
    elif transaction_lower in ['maturity']:
        category = "Principal"
        
        # Step 1: Debit Memo (Maturity Recognition)
        journal_entries.append({
            "journal_type": "Debit Memo",
            "subtype_code": "MAT",
            "description": f"{transaction_type} - Maturity Recognition",
            "entries": [
                {
                    "account_class": "Investment Receivable",
                    "account_subclass": "Investment Receivable", 
                    "account_type": "DR",
                    "formula_expression": "PRINCIPAL_AMOUNT",
                    "weight": 1
                },
                {
                    "account_class": "Marketable Securities",
                    "account_subclass": "Investment - Time Deposit" if asset_class == "6" else "Investment - Fixed Income",
                    "account_type": "CR",
                    "formula_expression": "BOOK_VALUE",
                    "weight": 2
                },
                {
                    "account_class": "Interest Receivable",
                    "account_subclass": "Interest Receivable",
                    "account_type": "DR", 
                    "formula_expression": "ACCRUED_INTEREST",
                    "weight": 3
                },
                {
                    "account_class": "Investment in Financial Instrument",
                    "account_subclass": "Investment - Interest Income",
                    "account_type": "CR",
                    "formula_expression": "ACCRUED_INTEREST",
                    "weight": 4
                }
            ]
        })
        
        # Step 2: Refund Created (Cash Receipt)
        journal_entries.append({
            "journal_type": "Refund Created",
            "subtype_code": "MAT_RC", 
            "description": f"{transaction_type} - Cash Receipt",
            "entries": [
                {
                    "account_class": "Cash for Investment",
                    "account_subclass": "Cash for Investment",
                    "account_type": "DR",
                    "formula_expression": "TOTAL_PROCEEDS",
                    "weight": 1
                },
                {
                    "account_class": "Investment Receivable",
                    "account_subclass": "Investment Receivable",
                    "account_type": "CR", 
                    "formula_expression": "PRINCIPAL_AMOUNT",
                    "weight": 2
                },
                {
                    "account_class": "Interest Receivable",
                    "account_subclass": "Interest Receivable",
                    "account_type": "CR",
                    "formula_expression": "ACCRUED_INTEREST", 
                    "weight": 3
                }
            ]
        })
        
    elif transaction_lower in ['placement']:
        category = "Investment"
        
        # Step 1: Invoice (Initial Recognition)
        journal_entries.append({
            "journal_type": "Invoice",
            "subtype_code": "PLA",
            "description": f"{transaction_type} - Initial Recognition",
            "entries": [
                {
                    "account_class": "Marketable Securities",
                    "account_subclass": "Investment - Time Deposit",
                    "account_type": "DR",
                    "formula_expression": "AMOUNT",
                    "weight": 1
                },
                {
                    "account_class": "Other payable",
                    "account_subclass": "Other payable",
                    "account_type": "CR",
                    "formula_expression": "AMOUNT",
                    "weight": 2
                }
            ]
        })
        
        # Step 2: Payment Created (Cash Settlement) 
        journal_entries.append({
            "journal_type": "Payment Created",
            "subtype_code": "PLA_PC",
            "description": f"{transaction_type} - Cash Settlement",
            "entries": [
                {
                    "account_class": "Other payable",
                    "account_subclass": "Other payable",
                    "account_type": "DR",
                    "formula_expression": "AMOUNT",
                    "weight": 1
                },
                {
                    "account_class": "Cash for Investment",
                    "account_subclass": "Cash for Investment", 
                    "account_type": "CR",
                    "formula_expression": "AMOUNT",
                    "weight": 2
                }
            ]
        })
        
    else:
        # Default setup for unknown transactions
        category = "Trading"
        journal_entries.append({
            "journal_type": "Invoice",
            "subtype_code": "UNK",
            "description": f"{transaction_type} - Generic Entry",
            "entries": [
                {
                    "account_class": "Investment in Financial Instrument",
                    "account_subclass": "Investment - Other",
                    "account_type": "DR",
                    "formula_expression": "AMOUNT",
                    "weight": 1
                },
                {
                    "account_class": "Cash for Investment",
                    "account_subclass": "Cash for Investment",
                    "account_type": "CR", 
                    "formula_expression": "AMOUNT",
                    "weight": 2
                }
            ]
        })
    
    # Get asset class information
    asset_info = {}
    if asset_class and asset_class in mapping.get("asset_classes", {}):
        asset_info = mapping["asset_classes"][asset_class]
    
    # Create complete journal setup
    suggestion = {
        "header": {
            "transaction_type": transaction_type,
            "asset_class": asset_class,
            "asset_class_name": asset_info.get("name", "Unknown"),
            "transaction_category": category,
            "company_id": 1,  # Default company
            "is_interface": "Y",
            "weight": 1
        },
        "journal_entries": journal_entries,
        "reasoning": {
            "category_reason": f"Transaction '{transaction_type}' classified as '{category}' based on Master Journal patterns",
            "structure_reason": f"Generated {len(journal_entries)} journal entries following Master Journal.csv structure",
            "settlement_reason": "Includes both recognition (Invoice/Debit Memo) and settlement (Payment/Refund Created) entries",
            "master_journal_reference": "Based on actual patterns found in Master Journal.csv data"
        },
        "confidence": "high",  # Based on actual Master Journal data
        "definition_context": definition_result.get("summary", ""),
        "suggested_at": datetime.now().isoformat()
    }
    
    return suggestion

def display_journal_suggestion(suggestion: Dict[str, Any]):
    """Display complete journal setup suggestion in a nice format"""
    
    st.subheader("📋 Complete Journal Setup")
    st.markdown(f"**Based on Master Journal.csv patterns** - {len(suggestion['journal_entries'])} journal entries")
    
    # Header information
    with st.expander("📊 Transaction Header", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Transaction Type:** {suggestion['header']['transaction_type']}")
            st.write(f"**Asset Class:** {suggestion['header']['asset_class_name']} ({suggestion['header']['asset_class']})")
            st.write(f"**Category:** {suggestion['header']['transaction_category']}")
        
        with col2:
            st.write(f"**Total Journal Entries:** {len(suggestion['journal_entries'])}")
            st.write(f"**Company ID:** {suggestion['header']['company_id']}")
            st.write(f"**Interface:** {suggestion['header']['is_interface']}")
    
    # Display each journal entry
    for i, journal_entry in enumerate(suggestion['journal_entries'], 1):
        with st.expander(f"📝 Journal Entry {i}: {journal_entry['journal_type']} ({journal_entry['subtype_code']})", expanded=True):
            st.write(f"**Description:** {journal_entry['description']}")
            st.write(f"**Journal Type:** {journal_entry['journal_type']}")
            st.write(f"**Subtype Code:** {journal_entry['subtype_code']}")
            
            # Create entries table
            entries_data = []
            for entry in journal_entry['entries']:
                entries_data.append({
                    "Account Class": entry['account_class'],
                    "Account Subclass": entry['account_subclass'],
                    "Type": entry['account_type'],
                    "Formula": entry['formula_expression'],
                    "Weight": entry['weight']
                })
            
            entries_df = pd.DataFrame(entries_data)
            st.dataframe(entries_df, use_container_width=True, hide_index=True)
    
    # Summary table of all entries
    with st.expander("📊 Complete Journal Summary", expanded=False):
        all_entries = []
        for i, journal_entry in enumerate(suggestion['journal_entries'], 1):
            for j, entry in enumerate(journal_entry['entries'], 1):
                all_entries.append({
                    "Step": f"{i}. {journal_entry['journal_type']}",
                    "Entry": f"{i}.{j}",
                    "Account Class": entry['account_class'],
                    "Account Subclass": entry['account_subclass'],
                    "Type": entry['account_type'],
                    "Formula": entry['formula_expression']
                })
        
        summary_df = pd.DataFrame(all_entries)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Reasoning
    with st.expander("🤔 Suggestion Reasoning"):
        st.write("**Category Selection:**")
        st.write(suggestion['reasoning']['category_reason'])
        
        st.write("**Journal Structure:**")
        st.write(suggestion['reasoning']['structure_reason'])
        
        st.write("**Settlement Pattern:**")
        st.write(suggestion['reasoning']['settlement_reason'])
        
        st.write("**Reference:**")
        st.write(suggestion['reasoning']['master_journal_reference'])
        
        st.info(f"**Confidence Level:** {suggestion['confidence'].title()}")
    
    # Show pattern examples from Master Journal
    with st.expander("📚 Master Journal Examples"):
        transaction_lower = suggestion['header']['transaction_type'].lower()
        
        if transaction_lower in ['buy', 'purchase']:
            st.write("**Master Journal Pattern:**")
            st.code("""
Example from Master Journal.csv:
Row 8: Buy (BUY) - Invoice
  DR: Marketable Securities - Investment Fixed Income
  DR: Interest Receivable - Interest Receivable Bought  
  CR: Other payable

Row 11: Buy (BUY_PC) - Payment Created
  DR: Other payable
  CR: Cash for Investment
            """)
        elif transaction_lower in ['sell']:
            st.write("**Master Journal Pattern:**")
            st.code("""
Example from Master Journal.csv:  
Row 211: Sell (SELL) - Debit Memo
  DR: Investment Receivable
  CR: Other income
  CR: Marketable Securities - Investment Fixed Income

Row 217: Sell (SELL_RC) - Refund Created
  DR: Cash for Investment
  CR: Investment Receivable
            """)
        elif transaction_lower in ['dividend', 'dividends']:
            st.write("**Master Journal Pattern:**")
            st.code("""
Example from Master Journal.csv:
Row 167: Dividends (DVD) - Debit Memo
  DR: Investment Receivable
  CR: Investment in Financial Instrument - Investment Dividend

Row 169: Dividends (DVD_RC) - Refund Created  
  DR: Cash for Investment
  CR: Investment Receivable
            """)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save to Journal Setups"):
            # Add to journal setups session state
            setup_key = f"{suggestion['header']['transaction_type']}_{suggestion['header']['asset_class']}"
            if "journal_setups" not in st.session_state:
                st.session_state.journal_setups = {}
            st.session_state.journal_setups[setup_key] = suggestion
            st.success("✅ Saved to Journal Setups!")
    
    with col2:
        if st.button("📋 Copy to Journal Setup Page"):
            st.info("💡 Switch to the Journal Setup page to use this configuration")
    
    with col3:
        if st.button("🔄 Modify Suggestion"):
            st.info("💡 Use the Journal Setup page to customize this suggestion")


# Page configuration
st.set_page_config(
    page_title="Definition Search - TReA",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📖 Transaction Definition Search")
st.markdown("Search and manage transaction definitions using Brave Search API")

# Initialize Brave Search processor
@st.cache_resource
def get_brave_processor():
    """Initialize Brave Search processor with caching"""
    try:
        if settings.brave_search_enabled and settings.brave_api_key:
            return BraveSearchProcessor(settings.brave_api_key)
        else:
            return None
    except Exception:
        return None

brave_processor = get_brave_processor()

# Initialize session state
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "definition_cache" not in st.session_state:
    st.session_state.definition_cache = {}
if "favorite_definitions" not in st.session_state:
    st.session_state.favorite_definitions = {}

# Sidebar
with st.sidebar:
    st.header("🔧 Search Configuration")
    
    # Service status
    if brave_processor:
        st.success("✅ Brave Search Available")
        st.write(f"**API Key:** {'*' * 20}")
    else:
        st.error("❌ Brave Search Not Available")
        st.warning("Please configure BRAVE_API_KEY in .env file")
    
    st.divider()
    
    # Search options
    st.subheader("Search Options")
    
    search_context = st.selectbox(
        "Search Context",
        [
            "banking finance treasury",
            "investment trading securities",
            "accounting journal entries",
            "financial instruments",
            "portfolio management"
        ],
        index=0
    )
    
    max_results = st.slider(
        "Max Results",
        min_value=3,
        max_value=20,
        value=5
    )
    
    include_sources = st.checkbox(
        "Include Source URLs",
        value=True
    )
    
    auto_cache = st.checkbox(
        "Auto-cache Results",
        value=True
    )
    
    st.divider()
    
    # Quick stats
    st.subheader("📊 Quick Stats")
    
    st.metric("Search History", len(st.session_state.search_history))
    st.metric("Cached Definitions", len(st.session_state.definition_cache))
    st.metric("Favorites", len(st.session_state.favorite_definitions))
    
    # Clear data options
    if st.button("🗑️ Clear History"):
        st.session_state.search_history = []
        st.rerun()
    
    if st.button("🗑️ Clear Cache"):
        st.session_state.definition_cache = {}
        st.rerun()

# Check if Brave Search is available
if not brave_processor:
    st.error("🚫 Brave Search is not configured")
    st.markdown("""
    **To enable Brave Search:**
    1. Get a free API key from [Brave Search API](https://brave.com/search/api/)
    2. Add `BRAVE_API_KEY=your_api_key` to your `.env` file
    3. Restart the application
    
    **Features available with Brave Search:**
    - Transaction type definitions
    - Financial terminology explanations
    - Contextual search results
    - Automatic definition caching
    """)
    st.stop()

# Main content
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 Single Search", "📊 Batch Search", "🆕 New Transactions", "📚 Browse Definitions", "⭐ Favorites", "📈 Analytics"])

# Tab 1: Single Search
with tab1:
    st.header("🔍 Single Transaction Search")
    st.markdown("Search for definitions of specific transaction types")
    
    # Search form
    with st.form("single_search_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_type = st.text_input(
                "Transaction Type",
                placeholder="e.g., PURCHASE, DIVIDEND, SETTLEMENT",
                help="Enter the transaction type to search for"
            )
        
        with col2:
            asset_class = st.text_input(
                "Asset Class (Optional)",
                placeholder="e.g., STOCK, BOND, CASH",
                help="Optional: Asset class for more specific context"
            )
        
        search_submitted = st.form_submit_button("🔍 Search Definition", type="primary")
    
    if search_submitted and transaction_type:
        # Check cache first
        asset_class_clean = (asset_class or "").strip()
        cache_key = f"{transaction_type.strip()}_{asset_class_clean}_{search_context}"
        
        # Debug info
        st.write(f"Debug: Cache key = `{cache_key}`")
        st.write(f"Debug: Cache contains {len(st.session_state.definition_cache)} items")
        if st.session_state.definition_cache:
            st.write("Debug: Cache keys:", list(st.session_state.definition_cache.keys()))
        
        # Add force new search option outside the main flow
        col1, col2 = st.columns(2)
        with col1:
            force_new = st.checkbox("🔄 Force New Search (ignore cache)", value=False)
        with col2:
            if st.button("🗑️ Clear Cache for This Search"):
                if cache_key in st.session_state.definition_cache:
                    del st.session_state.definition_cache[cache_key]
                    st.success("🗑️ Cache cleared for this search!")
                    st.rerun()
        
        if cache_key in st.session_state.definition_cache and not force_new:
            st.info("📋 Loading from cache...")
            search_result = st.session_state.definition_cache[cache_key]
            st.success("✅ Loaded from cache successfully!")
        else:
            # Perform new search
            with st.spinner(f"🔍 Searching for '{transaction_type}' definition..."):
                search_result = search_transaction_definition_sync(
                    transaction_type=transaction_type,
                    asset_class=asset_class or "",
                    api_key=settings.brave_api_key
                )
            
            # Cache result if auto-cache is enabled
            if auto_cache and search_result.get("success"):
                st.session_state.definition_cache[cache_key] = search_result
        
        # Display results
        if search_result.get("success"):
            st.success("✅ Definition found!")
            
            # Summary
            st.subheader("📄 Summary")
            st.write(search_result.get("summary", "No summary available"))
            
            # Definitions
            if search_result.get("definitions"):
                st.subheader("📖 Detailed Definitions")
                
                for i, definition in enumerate(search_result["definitions"][:max_results]):
                    with st.expander(f"📑 Definition {i+1}: {definition['title']}"):
                        st.write(f"**Relevance Score:** {definition['relevance_score']:.2f}")
                        st.write(f"**Description:** {definition['description']}")
                        
                        if include_sources:
                            st.write(f"🔗 **Source:** [{definition['title']}]({definition['url']})")
                        
                        # Add to favorites button
                        if st.button(f"⭐ Add to Favorites", key=f"fav_{i}_{transaction_type}"):
                            fav_key = f"{transaction_type}_{asset_class}_{i}"
                            st.session_state.favorite_definitions[fav_key] = {
                                "transaction_type": transaction_type,
                                "asset_class": asset_class,
                                "definition": definition,
                                "added_at": datetime.now().isoformat()
                            }
                            st.success("⭐ Added to favorites!")
                            st.rerun()
            
            # All sources
            if search_result.get("sources") and include_sources:
                with st.expander("🔗 All Sources"):
                    for i, source in enumerate(search_result["sources"][:10]):
                        st.write(f"{i+1}. [{source['title']}]({source['url']})")
                        st.write(f"   {source['description'][:200]}...")
                        st.divider()
            
            # Generate journal setup suggestion
            st.divider()
            try:
                journal_suggestion = suggest_journal_setup(transaction_type, asset_class_clean, search_result)
                display_journal_suggestion(journal_suggestion)
            except Exception as e:
                st.error(f"❌ Error generating journal suggestion: {str(e)}")
            
            # Add to search history
            history_entry = {
                "transaction_type": transaction_type,
                "asset_class": asset_class,
                "search_context": search_context,
                "searched_at": datetime.now().isoformat(),
                "definitions_found": len(search_result.get("definitions", [])),
                "success": True
            }
            st.session_state.search_history.append(history_entry)
        
        else:
            st.error(f"❌ Search failed: {search_result.get('error', 'Unknown error')}")
            
            # Add failed search to history
            history_entry = {
                "transaction_type": transaction_type,
                "asset_class": asset_class,
                "search_context": search_context,
                "searched_at": datetime.now().isoformat(),
                "definitions_found": 0,
                "success": False,
                "error": search_result.get("error", "Unknown error")
            }
            st.session_state.search_history.append(history_entry)

# Tab 2: Batch Search
with tab2:
    st.header("📊 Batch Transaction Search")
    st.markdown("Search for definitions of multiple transaction types at once")
    
    # Upload CSV for batch search
    st.subheader("📤 Upload Transaction Pairs")
    
    # CSV upload
    uploaded_file = st.file_uploader(
        "Upload CSV with transaction pairs",
        type=['csv'],
        help="CSV should have 'transaction_type' and 'asset_class' columns"
    )
    
    batch_pairs = []
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'transaction_type' in df.columns:
                batch_pairs = []
                for _, row in df.iterrows():
                    batch_pairs.append({
                        'transaction_type': str(row['transaction_type']).strip(),
                        'asset_class': str(row.get('asset_class', '')).strip()
                    })
                
                st.success(f"✅ Loaded {len(batch_pairs)} transaction pairs")
                st.dataframe(df.head(10), use_container_width=True)
            else:
                st.error("❌ CSV must contain 'transaction_type' column")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")
    
    # Manual batch entry
    st.divider()
    st.subheader("✏️ Manual Batch Entry")
    
    manual_pairs_text = st.text_area(
        "Enter transaction pairs (one per line, format: transaction_type,asset_class)",
        placeholder="PURCHASE,STOCK\nSALE,BOND\nDIVIDEND,EQUITY",
        height=100
    )
    
    if manual_pairs_text:
        manual_pairs = []
        for line in manual_pairs_text.strip().split('\n'):
            if ',' in line:
                parts = line.split(',')
                manual_pairs.append({
                    'transaction_type': parts[0].strip(),
                    'asset_class': parts[1].strip() if len(parts) > 1 else ''
                })
        
        if manual_pairs:
            batch_pairs.extend(manual_pairs)
            st.info(f"📝 Added {len(manual_pairs)} manual pairs")
    
    # Perform batch search
    if batch_pairs and st.button("🚀 Start Batch Search", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        try:
            status_text.text("🔍 Starting batch search...")
            
            # Use the batch search function
            batch_results = search_multiple_transactions_sync(
                batch_pairs,
                settings.brave_api_key
            )
            
            progress_bar.progress(1.0)
            status_text.text("✅ Batch search completed!")
            
            # Display results
            with results_container:
                if batch_results.get("success"):
                    st.success(f"✅ Batch search completed!")
                    st.write(f"**Total Searched:** {batch_results['total_searched']}")
                    st.write(f"**Definitions Found:** {batch_results['definitions_found']}")
                    
                    if batch_results.get("errors"):
                        st.warning(f"⚠️ {len(batch_results['errors'])} searches failed")
                    
                    # Display individual results
                    st.subheader("📋 Individual Results")
                    
                    for pair_key, result in batch_results.get("transactions", {}).items():
                        if result.get("success") and result.get("definitions"):
                            with st.expander(f"📖 {result['transaction_type']} - {result['asset_class']}"):
                                st.write(f"**Summary:** {result.get('summary', 'No summary')}")
                                
                                # Show top definitions
                                for i, defn in enumerate(result["definitions"][:3]):
                                    st.write(f"**{i+1}.** {defn['description'][:200]}...")
                                    if include_sources:
                                        st.write(f"🔗 [Source]({defn['url']})")
                                    st.divider()
                                
                                # Cache result
                                if auto_cache:
                                    cache_key = f"{result['transaction_type']}_{result['asset_class']}_{search_context}"
                                    st.session_state.definition_cache[cache_key] = result
                    
                    # Show errors if any
                    if batch_results.get("errors"):
                        with st.expander("❌ Failed Searches"):
                            for error in batch_results["errors"]:
                                st.write(f"**{error['transaction_type']} - {error['asset_class']}:** {error['error']}")
                
                else:
                    st.error(f"❌ Batch search failed: {batch_results.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"❌ Error during batch search: {str(e)}")
            progress_bar.progress(0)
            status_text.text("❌ Search failed")

# Tab 3: New Transactions
with tab3:
    st.header("🆕 New Transaction Search")
    st.markdown("Automatically detect and search definitions for new transactions from your processed data")
    
    # Check if we have transaction pairs from Vector Embeddings
    if "transaction_pairs" in st.session_state and st.session_state.transaction_pairs:
        transaction_pairs = st.session_state.transaction_pairs
        
        # Find new transactions (those without cached definitions)
        new_transactions = []
        for pair in transaction_pairs:
            cache_key = f"{pair['transaction_type']}_{pair.get('asset_class_id', pair.get('asset_class', 'unknown'))}"
            if cache_key not in st.session_state.definition_cache:
                new_transactions.append(pair)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📊 Total Transactions", len(transaction_pairs))
        with col2:
            st.metric("🆕 New Transactions", len(new_transactions))
        
        if new_transactions:
            st.subheader("🔍 New Transactions Detected")
            st.markdown("These transactions don't have definitions yet. Search for them automatically!")
            
            # Display new transactions in a table
            new_df = pd.DataFrame([
                {
                    "Transaction Type": pair["transaction_type"],
                    "Asset Class ID": pair.get("asset_class_id", "N/A"),
                    "Asset Class": pair.get("asset_class", "N/A"),
                    "Subtype": pair.get("transaction_subtype", "N/A"),
                    "Code": pair.get("transaction_subtype_code", "N/A")
                }
                for pair in new_transactions
            ])
            
            st.dataframe(new_df, use_container_width=True)
            
            # Search options for new transactions
            st.subheader("🚀 Search Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Search all new transactions
                if st.button("🔍 Search All New Transactions", type="primary"):
                    search_all_new_transactions(new_transactions, search_context, max_results)
            
            with col2:
                # Select specific transactions to search
                selected_transactions = st.multiselect(
                    "Select transactions to search",
                    options=range(len(new_transactions)),
                    format_func=lambda x: f"{new_transactions[x]['transaction_type']} - {new_transactions[x].get('asset_class', 'N/A')}",
                    default=[]
                )
                
                if selected_transactions and st.button("🔍 Search Selected"):
                    selected_pairs = [new_transactions[i] for i in selected_transactions]
                    search_all_new_transactions(selected_pairs, search_context, max_results)
            
            # Individual transaction search
            st.subheader("🎯 Individual Transaction Search")
            
            selected_transaction = st.selectbox(
                "Choose a transaction to search",
                options=range(len(new_transactions)),
                format_func=lambda x: f"{new_transactions[x]['transaction_type']} - {new_transactions[x].get('asset_class', 'N/A')}"
            )
            
            if st.button("🔍 Search This Transaction"):
                pair = new_transactions[selected_transaction]
                search_single_new_transaction(pair, search_context, max_results)
        
        else:
            st.success("✅ All your transactions already have definitions!")
            st.info("💡 Upload new CSV data in the Vector Embeddings page to find more transactions.")
    
    else:
        st.info("📊 No transaction data found")
        st.markdown("""
        **To search for new transactions:**
        1. Go to the **Vector Embeddings** page
        2. Upload your CSV with transaction data
        3. Process the CSV data
        4. Return here to search for definitions
        
        The system will automatically detect transactions that don't have definitions yet.
        """)

# Tab 4: Browse Definitions
with tab4:
    st.header("📚 Browse Cached Definitions")
    
    if not st.session_state.definition_cache:
        st.info("📭 No cached definitions found. Perform some searches first!")
    else:
        # Search through cached definitions
        st.subheader("🔍 Search Cached Definitions")
        
        search_term = st.text_input(
            "Search in cached definitions",
            placeholder="Enter keywords to search..."
        )
        
        # Filter definitions
        filtered_definitions = {}
        
        for cache_key, definition_data in st.session_state.definition_cache.items():
            if not search_term:
                filtered_definitions[cache_key] = definition_data
            else:
                # Search in transaction type, asset class, and summary
                search_text = f"{definition_data.get('transaction_type', '')} {definition_data.get('asset_class', '')} {definition_data.get('summary', '')}".lower()
                if search_term.lower() in search_text:
                    filtered_definitions[cache_key] = definition_data
        
        if filtered_definitions:
            st.write(f"📋 Showing {len(filtered_definitions)} cached definitions")
            
            # Display definitions
            for cache_key, definition_data in filtered_definitions.items():
                transaction_type = definition_data.get('transaction_type', 'Unknown')
                asset_class = definition_data.get('asset_class', 'Unknown')
                
                with st.expander(f"📖 {transaction_type} - {asset_class}"):
                    # Basic info
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Transaction Type:** {transaction_type}")
                        st.write(f"**Asset Class:** {asset_class}")
                        st.write(f"**Searched At:** {definition_data.get('searched_at', 'Unknown')}")
                    
                    with col2:
                        definitions_count = len(definition_data.get('definitions', []))
                        st.write(f"**Definitions Found:** {definitions_count}")
                        st.write(f"**Sources Count:** {len(definition_data.get('sources', []))}")
                    
                    # Summary
                    st.write("**Summary:**")
                    st.write(definition_data.get('summary', 'No summary available'))
                    
                    # Top definitions
                    if definition_data.get('definitions'):
                        st.write("**Top Definitions:**")
                        for i, defn in enumerate(definition_data['definitions'][:3]):
                            st.write(f"{i+1}. {defn['description'][:150]}...")
                    
                    # Actions
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button(f"⭐ Favorite", key=f"fav_cache_{cache_key}"):
                            fav_key = f"cache_{cache_key}"
                            st.session_state.favorite_definitions[fav_key] = {
                                "transaction_type": transaction_type,
                                "asset_class": asset_class,
                                "definition_data": definition_data,
                                "added_at": datetime.now().isoformat(),
                                "source": "cache"
                            }
                            st.success("⭐ Added to favorites!")
                            st.rerun()
                    
                    with col2:
                        if st.button(f"🔄 Refresh", key=f"refresh_{cache_key}"):
                            # Remove from cache to force new search
                            del st.session_state.definition_cache[cache_key]
                            st.info("🔄 Removed from cache. Search again to refresh.")
                            st.rerun()
                    
                    with col3:
                        if st.button(f"🗑️ Remove", key=f"remove_{cache_key}"):
                            del st.session_state.definition_cache[cache_key]
                            st.success("🗑️ Removed from cache")
                            st.rerun()
        else:
            st.info("🔍 No definitions match your search criteria")

# Tab 5: Favorites
with tab5:
    st.header("⭐ Favorite Definitions")
    
    if not st.session_state.favorite_definitions:
        st.info("⭐ No favorite definitions yet. Add some from your search results!")
    else:
        st.write(f"📋 You have {len(st.session_state.favorite_definitions)} favorite definitions")
        
        # Display favorites
        for fav_key, fav_data in st.session_state.favorite_definitions.items():
            transaction_type = fav_data.get('transaction_type', 'Unknown')
            asset_class = fav_data.get('asset_class', 'Unknown')
            added_at = fav_data.get('added_at', 'Unknown')
            
            with st.expander(f"⭐ {transaction_type} - {asset_class}"):
                st.write(f"**Added:** {added_at[:19] if added_at != 'Unknown' else 'Unknown'}")
                st.write(f"**Source:** {fav_data.get('source', 'search')}")
                
                # Display definition content
                if 'definition' in fav_data:
                    # Single definition
                    definition = fav_data['definition']
                    st.write(f"**Title:** {definition['title']}")
                    st.write(f"**Description:** {definition['description']}")
                    st.write(f"**Relevance:** {definition['relevance_score']:.2f}")
                    if include_sources:
                        st.write(f"🔗 [Source]({definition['url']})")
                
                elif 'definition_data' in fav_data:
                    # Full definition data
                    definition_data = fav_data['definition_data']
                    st.write(f"**Summary:** {definition_data.get('summary', 'No summary')}")
                    
                    if definition_data.get('definitions'):
                        st.write("**Top Definition:**")
                        top_def = definition_data['definitions'][0]
                        st.write(top_def['description'][:300] + "...")
                
                # Remove from favorites
                if st.button(f"🗑️ Remove from Favorites", key=f"remove_fav_{fav_key}"):
                    del st.session_state.favorite_definitions[fav_key]
                    st.success("🗑️ Removed from favorites")
                    st.rerun()
        
        # Export favorites
        st.divider()
        
        if st.button("📤 Export Favorites"):
            export_data = {
                "favorites": st.session_state.favorite_definitions,
                "exported_at": datetime.now().isoformat(),
                "total_count": len(st.session_state.favorite_definitions)
            }
            
            json_data = json.dumps(export_data, indent=2)
            st.download_button(
                label="💾 Download JSON",
                data=json_data,
                file_name=f"favorite_definitions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Tab 6: Analytics
with tab6:
    st.header("📈 Search Analytics")
    
    if not st.session_state.search_history:
        st.info("📊 No search history available. Perform some searches first!")
    else:
        # Basic metrics
        total_searches = len(st.session_state.search_history)
        successful_searches = len([h for h in st.session_state.search_history if h.get('success')])
        success_rate = (successful_searches / total_searches * 100) if total_searches > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Searches", total_searches)
        with col2:
            st.metric("Successful", successful_searches)
        with col3:
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col4:
            st.metric("Cached Results", len(st.session_state.definition_cache))
        
        # Search history table
        st.subheader("📋 Search History")
        
        history_data = []
        for entry in st.session_state.search_history[-20:]:  # Last 20 searches
            history_data.append({
                "Transaction Type": entry.get('transaction_type', ''),
                "Asset Class": entry.get('asset_class', ''),
                "Context": entry.get('search_context', ''),
                "Success": "✅" if entry.get('success') else "❌",
                "Definitions": entry.get('definitions_found', 0),
                "Searched At": entry.get('searched_at', '')[:19] if entry.get('searched_at') else ''
            })
        
        if history_data:
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        # Charts
        if len(st.session_state.search_history) > 1:
            import plotly.express as px
            
            # Success rate over time
            st.subheader("📈 Search Trends")
            
            # Prepare data for plotting
            df_history = pd.DataFrame(st.session_state.search_history)
            df_history['searched_date'] = pd.to_datetime(df_history['searched_at']).dt.date
            
            # Success rate by date
            daily_stats = df_history.groupby('searched_date').agg({
                'success': ['count', 'sum']
            }).reset_index()
            
            daily_stats.columns = ['date', 'total_searches', 'successful_searches']
            daily_stats['success_rate'] = (daily_stats['successful_searches'] / daily_stats['total_searches'] * 100).round(1)
            
            if len(daily_stats) > 1:
                fig1 = px.line(
                    daily_stats,
                    x='date',
                    y='success_rate',
                    title='Search Success Rate Over Time',
                    labels={'success_rate': 'Success Rate (%)', 'date': 'Date'}
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            # Most searched transaction types
            transaction_counts = df_history['transaction_type'].value_counts().head(10)
            
            if len(transaction_counts) > 1:
                fig2 = px.bar(
                    x=transaction_counts.index,
                    y=transaction_counts.values,
                    title='Most Searched Transaction Types',
                    labels={'x': 'Transaction Type', 'y': 'Search Count'}
                )
                fig2.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig2, use_container_width=True)
        
        # Export analytics
        st.divider()
        
        if st.button("📤 Export Analytics Data"):
            analytics_data = {
                "search_history": st.session_state.search_history,
                "summary": {
                    "total_searches": total_searches,
                    "successful_searches": successful_searches,
                    "success_rate": success_rate,
                    "cached_results": len(st.session_state.definition_cache)
                },
                "exported_at": datetime.now().isoformat()
            }
            
            json_data = json.dumps(analytics_data, indent=2)
            st.download_button(
                label="💾 Download Analytics JSON",
                data=json_data,
                file_name=f"search_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


# Helper functions for new transaction search
def search_single_new_transaction(pair: Dict[str, Any], search_context: str, max_results: int):
    """Search definition for a single new transaction"""
    
    transaction_type = pair["transaction_type"]
    asset_class = pair.get("asset_class", pair.get("asset_class_id", "unknown"))
    
    with st.spinner(f"🔍 Searching definition for {transaction_type}..."):
        try:
            # Create search query
            search_query = f"{transaction_type} transaction {search_context}"
            
            # Search using Brave
            result = search_transaction_definition_sync(
                transaction_type=transaction_type,
                asset_class=str(asset_class),
                brave_processor=brave_processor,
                max_results=max_results
            )
            
            if result.get("success"):
                # Cache the result
                cache_key = f"{transaction_type}_{asset_class}"
                st.session_state.definition_cache[cache_key] = {
                    "transaction_type": transaction_type,
                    "asset_class": str(asset_class),
                    "searched_at": datetime.now().isoformat(),
                    **result
                }
                
                # Add to search history
                st.session_state.search_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "transaction_type": transaction_type,
                    "asset_class": str(asset_class),
                    "success": True,
                    "results_count": len(result.get("definitions", []))
                })
                
                st.success(f"✅ Found definition for {transaction_type}")
                
                # Display results
                with st.expander(f"📖 {transaction_type} Definition", expanded=True):
                    st.write("**Summary:**")
                    st.write(result.get("summary", "No summary available"))
                    
                    if result.get("definitions"):
                        st.write("**Definitions:**")
                        for i, definition in enumerate(result["definitions"][:3], 1):
                            st.write(f"{i}. {definition}")
                    
                    if result.get("sources"):
                        st.write("**Sources:**")
                        for source in result["sources"][:3]:
                            st.write(f"- [{source.get('title', 'Source')}]({source.get('url', '#')})")
                    
                    # Add journal suggestion for new transaction
                    st.divider()
                    try:
                        journal_suggestion = suggest_journal_setup(transaction_type, str(asset_class), result)
                        display_journal_suggestion(journal_suggestion)
                    except Exception as e:
                        st.error(f"❌ Error generating journal suggestion: {str(e)}")
                
            else:
                st.error(f"❌ Failed to find definition for {transaction_type}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Error searching {transaction_type}: {str(e)}")


def search_all_new_transactions(new_transactions: List[Dict[str, Any]], search_context: str, max_results: int):
    """Search definitions for all new transactions"""
    
    st.info(f"🚀 Starting batch search for {len(new_transactions)} transactions...")
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    successful_searches = 0
    failed_searches = 0
    
    for i, pair in enumerate(new_transactions):
        transaction_type = pair["transaction_type"]
        asset_class = pair.get("asset_class", pair.get("asset_class_id", "unknown"))
        
        # Update progress
        progress = (i + 1) / len(new_transactions)
        progress_bar.progress(progress)
        status_text.text(f"🔍 Searching {i+1}/{len(new_transactions)}: {transaction_type}")
        
        try:
            # Search using Brave
            result = search_transaction_definition_sync(
                transaction_type=transaction_type,
                asset_class=str(asset_class),
                brave_processor=brave_processor,
                max_results=max_results
            )
            
            if result.get("success"):
                # Cache the result
                cache_key = f"{transaction_type}_{asset_class}"
                st.session_state.definition_cache[cache_key] = {
                    "transaction_type": transaction_type,
                    "asset_class": str(asset_class),
                    "searched_at": datetime.now().isoformat(),
                    **result
                }
                
                # Add to search history
                st.session_state.search_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "transaction_type": transaction_type,
                    "asset_class": str(asset_class),
                    "success": True,
                    "results_count": len(result.get("definitions", []))
                })
                
                successful_searches += 1
                
                # Show results in container
                with results_container:
                    with st.expander(f"✅ {transaction_type} - {asset_class}"):
                        st.write("**Summary:**")
                        st.write(result.get("summary", "No summary available"))
                        
                        if result.get("definitions"):
                            st.write("**Top Definition:**")
                            st.write(result["definitions"][0])
                        
                        # Add compact journal suggestion
                        st.write("**📋 Complete Journal Setup:**")
                        try:
                            suggestion = suggest_journal_setup(transaction_type, str(asset_class), result)
                            st.write(f"- **Category:** {suggestion['header']['transaction_category']}")
                            st.write(f"- **Journal Entries:** {len(suggestion['journal_entries'])}")
                            
                            for i, entry in enumerate(suggestion['journal_entries'], 1):
                                st.write(f"  **{i}. {entry['journal_type']} ({entry['subtype_code']}):**")
                                for detail in entry['entries']:
                                    st.write(f"    - {detail['account_type']}: {detail['account_class']}")
                        except Exception as e:
                            st.write(f"❌ Error: {str(e)}")
            else:
                failed_searches += 1
                with results_container:
                    st.error(f"❌ Failed: {transaction_type} - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            failed_searches += 1
            with results_container:
                st.error(f"❌ Error searching {transaction_type}: {str(e)}")
    
    # Final status
    progress_bar.progress(1.0)
    status_text.text("✅ Batch search completed!")
    
    # Summary
    st.success(f"🎉 Batch search completed! Successfully searched {successful_searches} transactions, {failed_searches} failed.")
    
    if successful_searches > 0:
        st.info("💡 All successful searches have been cached. Check the 'Browse Definitions' tab to review them.")


# Footer
st.divider()
st.markdown("💡 **Tips:** Use specific transaction types and asset classes for better search results. Enable auto-caching to improve performance.") 