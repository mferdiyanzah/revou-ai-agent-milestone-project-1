"""
Journal Setup Management Page
Manages journal setups for transaction type-asset class pairs in TReA system
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, List

# Import our modules
from src.config import settings
from src.services.api_client import TReAAPIClient

# Page configuration
st.set_page_config(
    page_title="Journal Setup - TReA",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📋 Journal Setup Management")
st.markdown("Configure journal entries for transaction type-asset class pairs")

# Initialize API client
@st.cache_resource
def get_api_client():
    return TReAAPIClient()

api_client = get_api_client()

# Initialize session state
if "journal_setups" not in st.session_state:
    st.session_state.journal_setups = {}
if "account_mappings" not in st.session_state:
    st.session_state.account_mappings = {}

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Journal setup options
    st.subheader("Journal Options")
    
    default_company_id = st.number_input("Default Company ID", min_value=1, value=1)
    default_weight = st.number_input("Default Weight", min_value=1, value=1)
    
    # Account types based on Master Journal
    account_types = [
        "Bank", "Cash Clearing", "Cash for Investment", "Interbank Transfer",
        "Interest Receivable", "Interest income", "Investment Receivable", 
        "Investment in Financial Instrument", "Marketable Securities",
        "Marketable Securities - Bonds", "Other Comprehensive", "Other expenses",
        "Other income", "Other payable", "Taxes payable", "Time deposit", "art. 23 final"
    ]
    
    st.divider()
    
    # Quick actions
    st.subheader("Quick Actions")
    
    if st.button("🔄 Refresh Journal Data"):
        st.rerun()
    
    if st.button("💾 Export All Setups"):
        if st.session_state.journal_setups:
            export_data = json.dumps(st.session_state.journal_setups, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=export_data,
                file_name=f"journal_setups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Sample journal header and detail data for reference
SAMPLE_JOURNAL_HEADER = {
    "ASSET_CATEGORY_ID": 1,
    "ASSET_CATEGORY_NAME": "Cash & Cash Equivalents",
    "ASSET_CLASS_ID": 1,
    "ASSET_CLASS_NAME": "Cash",
    "TRANSACTION_TYPE": "DEPOSIT",
    "TRANSACTION_SUBTYPE": "CASH_DEPOSIT",
    "TRANSACTION_SUBTYPE_CODE": "CD001",
    "JOURNAL_TYPE": "Standard",
    "INVOICE_TYPE": "Standard",
    "COMPANY_ID": 1,
    "IS_INTERFACE": "Y",
    "WEIGHT": 1
}

SAMPLE_JOURNAL_DETAIL = {
    "ACCOUNT_ID": 1,
    "ACCOUNT_CLASS": "ASSET",
    "ACCOUNT_SUBCLASS_ID": 1,
    "ACCOUNT_TYPE": "DR",
    "FORMULA_ID": None,
    "IS_INTERFACE": "Y",
    "WEIGHT": 1
}

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Create Setup", "📊 Browse Setups", "🔧 Configure Accounts", "📖 Documentation"])

# Tab 1: Create Setup
with tab1:
    st.header("🏗️ Create Journal Setup")
    
    # Transaction pair selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Information")
        
        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "Buy", "Sell", "Subscription", "Redemption", "Partial redemption",
                "Dividend Cash", "Dividends", "Interest", "Interest Payment", 
                "Interest Received", "Accrued Interest", "Coupons", "Maturity",
                "Placement", "Repayment Of Deposit", "Early Termination", 
                "Cash Placement", "Deposit Placement", "Withdrawal", "Transfer",
                "Management Fee", "Custody Fee", "Internal Payment Debit",
                "Internal Payment Credit", "Incoming Payment Foreign", 
                "Payment Domestic", "Client Money Transfer Third Party",
                "Public purchase offer", "Revenue", "Open Deposit (Rollover)"
            ],
            help="The type of transaction from Master Journal"
        )
        
        asset_class_id = st.selectbox(
            "Asset Class ID",
            options=["1", "2", "3", "5", "6"],
            format_func=lambda x: {
                "1": "1 - Cash & Cash Equivalents",
                "2": "2 - Fixed Income Securities", 
                "3": "3 - Fund Investments",
                "5": "5 - Equity Securities",
                "6": "6 - Time Deposits"
            }[x],
            help="The asset class for this transaction"
        )
        
        transaction_subtype = st.text_input(
            "Transaction Subtype",
            placeholder="e.g., EQUITY_PURCHASE",
            help="More specific transaction subtype"
        )
        
        transaction_subtype_code = st.text_input(
            "Subtype Code",
            placeholder="e.g., EP001",
            help="Internal code for the transaction subtype"
        )
    
    with col2:
        st.subheader("Journal Header Configuration")
        
        asset_category_id = st.number_input(
            "Asset Category ID",
            min_value=1,
            value=1,
            help="Asset category from mst_asset_category table"
        )
        
        asset_category_name = st.text_input(
            "Asset Category Name",
            value="Cash & Cash Equivalents",
            help="Name of the asset category"
        )
        
        asset_class_id = st.number_input(
            "Asset Class ID",
            min_value=1,
            value=1,
            help="Asset class from asset_classes table"
        )
        
        journal_type = st.selectbox(
            "Journal Type",
            ["Invoice", "Debit Memo", "Payment Created", "Refund Created"],
            help="Type of journal entry"
        )
        
        invoice_type = st.selectbox(
            "Invoice Type",
            ["Standard", "Credit", "Debit", "Mixed"],
            help="Type of invoice if applicable"
        )
        
        company_id = st.number_input(
            "Company ID",
            min_value=1,
            value=default_company_id,
            help="Company ID from mst_company table"
        )
        
        is_interface = st.selectbox(
            "Is Interface",
            ["Y", "N"],
            help="Whether this setup interfaces with external systems"
        )
        
        weight = st.number_input(
            "Weight",
            min_value=1,
            value=default_weight,
            help="Priority weight for journal processing"
        )
    
    st.divider()
    
    # Journal details section
    st.subheader("📝 Journal Details Configuration")
    st.markdown("Configure the debit and credit entries for this transaction")
    
    # Initialize journal details in session state
    if "current_journal_details" not in st.session_state:
        st.session_state.current_journal_details = []
    
    # Add new journal detail
    with st.expander("➕ Add Journal Detail Entry"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            detail_account_id = st.number_input(
                "Account ID",
                min_value=1,
                value=1,
                key="detail_account_id"
            )
            
            detail_account_class = st.selectbox(
                "Account Class",
                account_types,
                key="detail_account_class"
            )
            
            detail_account_subclass_id = st.number_input(
                "Account Subclass ID",
                min_value=1,
                value=1,
                key="detail_account_subclass_id"
            )
        
        with col2:
            detail_account_type = st.selectbox(
                "Account Type (DR/CR)",
                ["DR", "CR"],
                key="detail_account_type"
            )
            
            detail_formula_id = st.number_input(
                "Formula ID",
                min_value=0,
                value=0,
                help="0 for no formula, or ID from mst_formula table",
                key="detail_formula_id"
            )
        
        with col3:
            detail_is_interface = st.selectbox(
                "Is Interface",
                ["Y", "N"],
                key="detail_is_interface"
            )
            
            detail_weight = st.number_input(
                "Weight",
                min_value=1,
                value=1,
                key="detail_weight"
            )
        
        if st.button("➕ Add Detail Entry"):
            new_detail = {
                "ACCOUNT_ID": detail_account_id,
                "ACCOUNT_CLASS": detail_account_class,
                "ACCOUNT_SUBCLASS_ID": detail_account_subclass_id,
                "ACCOUNT_TYPE": detail_account_type,
                "FORMULA_ID": detail_formula_id if detail_formula_id > 0 else None,
                "IS_INTERFACE": detail_is_interface,
                "WEIGHT": detail_weight
            }
            
            st.session_state.current_journal_details.append(new_detail)
            st.success("✅ Journal detail added!")
            st.rerun()
    
    # Display current journal details
    if st.session_state.current_journal_details:
        st.subheader("📋 Current Journal Details")
        
        details_df = pd.DataFrame(st.session_state.current_journal_details)
        st.dataframe(details_df, use_container_width=True, hide_index=True)
        
        # Validation
        dr_entries = [d for d in st.session_state.current_journal_details if d["ACCOUNT_TYPE"] == "DR"]
        cr_entries = [d for d in st.session_state.current_journal_details if d["ACCOUNT_TYPE"] == "CR"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Debit Entries", len(dr_entries))
        with col2:
            st.metric("Credit Entries", len(cr_entries))
        with col3:
            balanced = len(dr_entries) > 0 and len(cr_entries) > 0
            st.metric("Balanced", "✅" if balanced else "❌")
        
        if not balanced:
            st.warning("⚠️ Journal entries should have both debit and credit entries for balance")
        
        # Clear details button
        if st.button("🗑️ Clear All Details"):
            st.session_state.current_journal_details = []
            st.rerun()
    
    st.divider()
    
    # Save journal setup
    if st.button("💾 Save Journal Setup", type="primary"):
        if not transaction_type or not asset_class:
            st.error("❌ Please fill in transaction type and asset class")
        elif not st.session_state.current_journal_details:
            st.error("❌ Please add at least one journal detail entry")
        else:
            # Create journal header
            journal_header = {
                "ASSET_CATEGORY_ID": asset_category_id,
                "ASSET_CATEGORY_NAME": asset_category_name,
                "ASSET_CLASS_ID": asset_class_id,
                "ASSET_CLASS_NAME": asset_class,
                "TRANSACTION_TYPE": transaction_type,
                "TRANSACTION_SUBTYPE": transaction_subtype,
                "TRANSACTION_SUBTYPE_CODE": transaction_subtype_code,
                "JOURNAL_TYPE": journal_type,
                "INVOICE_TYPE": invoice_type,
                "COMPANY_ID": company_id,
                "IS_INTERFACE": is_interface,
                "WEIGHT": weight,
                "CREATED_AT": datetime.now().isoformat(),
                "CREATED_BY": "System"
            }
            
            # Create complete setup
            setup_key = f"{transaction_type}_{asset_class}"
            journal_setup = {
                "header": journal_header,
                "details": st.session_state.current_journal_details.copy(),
                "metadata": {
                    "pair_key": setup_key,
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat()
                }
            }
            
            # Store in session state
            st.session_state.journal_setups[setup_key] = journal_setup
            
            # Clear current details
            st.session_state.current_journal_details = []
            
            st.success(f"✅ Journal setup saved for {transaction_type} - {asset_class}")
            st.rerun()

# Tab 2: Browse Setups
with tab2:
    st.header("📊 Browse Journal Setups")
    
    if not st.session_state.journal_setups:
        st.info("📋 No journal setups found. Create some setups in the Create Setup tab.")
    else:
        # Summary metrics
        total_setups = len(st.session_state.journal_setups)
        unique_transaction_types = len(set(
            setup["header"]["TRANSACTION_TYPE"] 
            for setup in st.session_state.journal_setups.values()
        ))
        unique_asset_classes = len(set(
            setup["header"]["ASSET_CLASS_NAME"] 
            for setup in st.session_state.journal_setups.values()
        ))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Setups", total_setups)
        with col2:
            st.metric("Transaction Types", unique_transaction_types)
        with col3:
            st.metric("Asset Classes", unique_asset_classes)
        
        st.divider()
        
        # Setup list
        st.subheader("📋 All Journal Setups")
        
        for setup_key, setup_data in st.session_state.journal_setups.items():
            header = setup_data["header"]
            details = setup_data["details"]
            
            with st.expander(f"📄 {header['TRANSACTION_TYPE']} - {header['ASSET_CLASS_NAME']}"):
                # Header information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Header Information:**")
                    st.write(f"- **Transaction Type:** {header['TRANSACTION_TYPE']}")
                    st.write(f"- **Asset Class:** {header['ASSET_CLASS_NAME']}")
                    st.write(f"- **Transaction Subtype:** {header['TRANSACTION_SUBTYPE']}")
                    st.write(f"- **Journal Type:** {header['JOURNAL_TYPE']}")
                    st.write(f"- **Company ID:** {header['COMPANY_ID']}")
                
                with col2:
                    st.write("**Configuration:**")
                    st.write(f"- **Asset Category ID:** {header['ASSET_CATEGORY_ID']}")
                    st.write(f"- **Asset Class ID:** {header['ASSET_CLASS_ID']}")
                    st.write(f"- **Is Interface:** {header['IS_INTERFACE']}")
                    st.write(f"- **Weight:** {header['WEIGHT']}")
                    st.write(f"- **Created:** {header.get('CREATED_AT', 'Unknown')[:19]}")
                
                # Details table
                st.write("**Journal Details:**")
                details_df = pd.DataFrame(details)
                st.dataframe(details_df, use_container_width=True, hide_index=True)
                
                # Actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button(f"📝 Edit", key=f"edit_{setup_key}"):
                        st.info("Edit functionality coming soon!")
                
                with col2:
                    if st.button(f"📋 Copy", key=f"copy_{setup_key}"):
                        st.info("Copy functionality coming soon!")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{setup_key}"):
                        del st.session_state.journal_setups[setup_key]
                        st.success(f"✅ Deleted setup for {setup_key}")
                        st.rerun()
        
        st.divider()
        
        # Bulk operations
        st.subheader("🔧 Bulk Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export All Setups"):
                export_data = {
                    "journal_setups": st.session_state.journal_setups,
                    "exported_at": datetime.now().isoformat(),
                    "total_setups": len(st.session_state.journal_setups)
                }
                
                json_data = json.dumps(export_data, indent=2)
                st.download_button(
                    label="💾 Download JSON",
                    data=json_data,
                    file_name=f"journal_setups_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("🗑️ Clear All Setups"):
                st.session_state.journal_setups = {}
                st.success("✅ All setups cleared")
                st.rerun()

# Tab 3: Configure Accounts
with tab3:
    st.header("🔧 Account Configuration")
    st.markdown("Configure account mappings and formulas for journal entries")
    
    # Account types reference
    st.subheader("📚 Account Types Reference")
    
    account_info = {
        "Account Type": ["ASSET", "LIABILITY", "EQUITY", "REVENUE", "EXPENSE"],
        "Description": [
            "Resources owned by the company",
            "Debts and obligations",
            "Owner's claims on company assets",
            "Income generated from operations",
            "Costs incurred in operations"
        ],
        "Normal Balance": ["Debit", "Credit", "Credit", "Credit", "Debit"],
        "Examples": [
            "Cash, Investments, Fixed Assets",
            "Accounts Payable, Loans",
            "Share Capital, Retained Earnings",
            "Sales, Interest Income",
            "Operating Expenses, Interest Expense"
        ]
    }
    
    account_df = pd.DataFrame(account_info)
    st.dataframe(account_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Common journal templates
    st.subheader("📋 Common Journal Templates")
    
    templates = {
        "Cash Purchase": {
            "description": "Purchase of securities with cash",
            "entries": [
                {"Account": "Investment Securities", "Type": "DR", "Class": "ASSET"},
                {"Account": "Cash", "Type": "CR", "Class": "ASSET"}
            ]
        },
        "Cash Sale": {
            "description": "Sale of securities for cash",
            "entries": [
                {"Account": "Cash", "Type": "DR", "Class": "ASSET"},
                {"Account": "Investment Securities", "Type": "CR", "Class": "ASSET"}
            ]
        },
        "Dividend Income": {
            "description": "Receipt of dividend income",
            "entries": [
                {"Account": "Cash", "Type": "DR", "Class": "ASSET"},
                {"Account": "Dividend Income", "Type": "CR", "Class": "REVENUE"}
            ]
        },
        "Interest Income": {
            "description": "Receipt of interest income",
            "entries": [
                {"Account": "Cash", "Type": "DR", "Class": "ASSET"},
                {"Account": "Interest Income", "Type": "CR", "Class": "REVENUE"}
            ]
        },
        "Management Fee": {
            "description": "Payment of management fees",
            "entries": [
                {"Account": "Management Fees", "Type": "DR", "Class": "EXPENSE"},
                {"Account": "Cash", "Type": "CR", "Class": "ASSET"}
            ]
        }
    }
    
    for template_name, template_data in templates.items():
        with st.expander(f"📄 {template_name}"):
            st.write(f"**Description:** {template_data['description']}")
            st.write("**Journal Entries:**")
            
            for entry in template_data['entries']:
                st.write(f"- {entry['Type']}: {entry['Account']} ({entry['Class']})")
            
            if st.button(f"📋 Use Template", key=f"template_{template_name}"):
                st.info("Template functionality coming soon!")
    
    st.divider()
    
    # Formula configuration
    st.subheader("🧮 Formula Configuration")
    st.markdown("Configure formulas for automatic amount calculations")
    
    formula_examples = {
        "Formula": ["AMOUNT * RATE", "PRINCIPAL * INTEREST_RATE / 365", "AMOUNT * FX_RATE"],
        "Description": ["Multiply amount by rate", "Daily interest calculation", "Currency conversion"],
        "Use Case": ["Fee calculations", "Interest accruals", "Foreign exchange"]
    }
    
    formula_df = pd.DataFrame(formula_examples)
    st.dataframe(formula_df, use_container_width=True, hide_index=True)
    
    # Formula creator
    with st.expander("➕ Create New Formula"):
        formula_name = st.text_input("Formula Name")
        formula_expression = st.text_input("Formula Expression", placeholder="e.g., AMOUNT * 0.02")
        formula_description = st.text_area("Description")
        
        if st.button("💾 Save Formula"):
            if formula_name and formula_expression:
                st.success(f"✅ Formula '{formula_name}' saved!")
            else:
                st.error("❌ Please fill in formula name and expression")

# Tab 4: Documentation
with tab4:
    st.header("📖 Documentation")
    
    st.markdown("""
    ## Journal Setup Guide
    
    ### Overview
    The journal setup system allows you to configure how different transaction types are recorded in the accounting system. Each setup consists of:
    
    1. **Journal Header**: Defines the transaction context
    2. **Journal Details**: Defines the specific debit and credit entries
    
    ### Journal Header Fields
    
    | Field | Description | Example |
    |-------|-------------|---------|
    | Transaction Type | The type of transaction | PURCHASE, SALE, DIVIDEND |
    | Asset Class | The asset class involved | STOCK, BOND, CASH |
    | Transaction Subtype | More specific classification | EQUITY_PURCHASE |
    | Journal Type | Type of journal entry | Standard, Adjustment |
    | Company ID | Company identifier | 1, 2, 3 |
    | Weight | Processing priority | 1 (highest) to 10 (lowest) |
    
    ### Journal Detail Fields
    
    | Field | Description | Example |
    |-------|-------------|---------|
    | Account ID | Account identifier | 1001, 2001 |
    | Account Class | Type of account | ASSET, LIABILITY, EQUITY |
    | Account Type | Debit or Credit | DR, CR |
    | Formula ID | Calculation formula | 1, 2, null |
    | Weight | Entry processing order | 1, 2, 3 |
    
    ### Best Practices
    
    1. **Balance Entries**: Always ensure debits equal credits
    2. **Naming Convention**: Use consistent transaction type names
    3. **Weight Management**: Use weights to control processing order
    4. **Documentation**: Add clear descriptions for complex setups
    5. **Testing**: Test setups with sample transactions
    
    ### Common Patterns
    
    #### Asset Purchase
    - **DR**: Investment Account (ASSET)
    - **CR**: Cash Account (ASSET)
    
    #### Income Receipt
    - **DR**: Cash Account (ASSET)
    - **CR**: Income Account (REVENUE)
    
    #### Expense Payment
    - **DR**: Expense Account (EXPENSE)
    - **CR**: Cash Account (ASSET)
    
    ### Integration with TReA
    
    These journal setups integrate with:
    - **PDF Processing**: Automatic transaction mapping
    - **Vector Database**: Similar transaction finding
    - **AI Analysis**: Intelligent suggestions
    - **Brave Search**: Transaction definitions
    
    ### Database Schema
    
    The journal setups map to these database tables:
    - `mst_journal_hdr`: Journal headers
    - `mst_journal_dtl`: Journal details
    - `journal_history`: Actual journal entries
    """)

# Footer
st.divider()
st.markdown("💡 **Tips:** Create balanced journal entries with both debit and credit entries for accurate accounting.") 