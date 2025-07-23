# 🏦 TReA - Treasury with Embedded AI

**Multimodal Agentic AI for Treasury Document Processing & Journal Mapping**

TReA is a comprehensive Streamlit application that automates treasury document processing using advanced AI capabilities. It processes **multiple input formats** including PDF statements, text files, JSON data, and CSV files from banks (like DBS Singapore), extracts transaction data, and automatically maps them to appropriate journal entries.

## 🎯 Features

- **📄 Multimodal Document Upload**: Upload treasury data in multiple formats (PDF, TXT, JSON, CSV)
- **🤖 Automated Processing**: AI-powered extraction and transformation of financial data
- **📊 Transaction Analysis**: Automatic categorization and mapping of transactions
- **🔍 Smart Definition Search**: Brave Search integration for transaction type definitions
- **📈 Analytics Dashboard**: Historical processing data and trends visualization
- **⚙️ System Monitoring**: Real-time health checks and configuration management
- **🔒 Secure**: Token-based authentication and file validation
- **📱 Responsive UI**: Modern, intuitive interface built with Streamlit

## 🏗️ Architecture

```
TReA Application
├── Frontend (Streamlit)
│   ├── Multimodal Document Upload
│   ├── Processing Dashboard
│   └── Analytics & Monitoring
├── Backend Services
│   ├── PDF Processing Pipeline (TReA API)
│   ├── Text/JSON/CSV Processing (Direct)
│   ├── AI/ML Models
│   └── Database Integration
└── TReA API Backend
    ├── PDF Parser Service
    ├── Data Transformation
    └── Journal Mapping
```

## 📁 Supported File Formats

### **📄 PDF Files**
- **Format**: Treasury statements (DBS Singapore format)
- **Processing**: Full TReA backend API integration
- **Features**: Complete PDF parsing, text extraction, and database integration

### **📝 Text Files (.txt)**
- **Format**: Plain text transaction data
- **Pattern**: `TRANSACTION_TYPE ASSET_CLASS AMOUNT CURRENCY DATE DESCRIPTION`
- **Example**: `BUY STOCK 1000 USD 2024-01-15 Purchase of equity`
- **Processing**: Direct regex-based parsing

### **📊 JSON Files (.json)**
- **Format**: Structured transaction data
- **Schema**: `{"transactions": [{"transaction_type": "BUY", "asset_class": "STOCK", ...}]}`
- **Processing**: Schema validation and object parsing
- **Features**: Flexible structure support

### **📈 CSV Files (.csv)**
- **Format**: Tabular transaction data
- **Columns**: transaction_type, asset_class, amount, currency, date, description
- **Processing**: Automatic column mapping and pandas integration
- **Features**: Header detection and encoding support

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- uv package manager (recommended) or pip
- Access to TReA backend API (for PDF processing)
- Valid API token
- Optional: OpenAI API key, Brave Search API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd milestone-project-1-ai-agent
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Configure environment**
   ```bash
   # Copy the environment template
   cp env.template .env
   
   # Edit .env with your configuration
   nano .env
   ```

4. **Run the application**
   ```bash
   # Using the main entry point (recommended)
   python main.py
   
   # Or using streamlit directly
   streamlit run main.py
   
   # Alternative: using uv
   uv run python main.py
   ```

5. **Access the application**
   - If using `python main.py`, the application will automatically launch in your browser
   - If using `streamlit run main.py`, navigate to `http://localhost:8501`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# API Configuration (Required for PDF processing)
API_BASE_URL=http://192.168.74.211:30477
API_TOKEN=your_bearer_token_here

# AI/ML Configuration (Optional - enhances all formats)
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small

# Brave Search Configuration (Optional - adds definition lookup)
BRAVE_API_KEY=your_brave_api_key_here
BRAVE_SEARCH_ENABLED=true

# Database Configuration (Optional - enables vector similarity)
POSTGRES_HOST=localhost
POSTGRES_DB=trea_vector_db
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

# File Upload Settings
MAX_FILE_SIZE_MB=50
UPLOAD_DIR=uploads

# Application Settings
DEBUG=false
```

### API Token Setup

1. Obtain a valid Bearer token from your TReA backend administrator (for PDF processing)
2. Add the token to your `.env` file as `API_TOKEN`
3. The token will be automatically used for PDF processing requests

## 📋 Usage Guide

### 1. **Document Processing**

#### **PDF Files** 
1. **Upload PDF**: Select a treasury statement PDF file
2. **Automatic Processing**: Full TReA backend integration
3. **Complete Pipeline**: Extract → Transform → Map → Store
4. **Database Integration**: Direct journal entry creation

#### **Text/JSON/CSV Files**
1. **Upload File**: Select your text, JSON, or CSV file
2. **Direct Processing**: No external API dependency
3. **AI Enhancement**: Definition lookup and analysis
4. **Quick Results**: Immediate processing and insights

### 2. **Processing Pipeline**

#### **PDF Processing Flow**
1. **File Upload** → Save to temporary location
2. **API Upload** → Send to TReA backend
3. **Content Extraction** → Extract text and structure from PDF
4. **Data Transformation** → Convert to structured transaction data
5. **Journal Mapping** → Map transactions to appropriate journal entries

#### **Text-based Processing Flow**
1. **File Upload** → Save and validate format
2. **Content Parsing** → Extract using format-specific parser
3. **Data Structuring** → Convert to standard transaction format
4. **AI Enhancement** → Add definitions and analysis
5. **Mock Mapping** → Create rule-based journal entries

### 3. **Analytics Dashboard**

- View processing success rates across all formats
- Analyze transaction type distributions
- Track historical processing data
- Export transaction data as CSV
- Source format indicators and metrics

### 4. **System Monitoring**

- Real-time API health checks
- Service availability for all processors
- Configuration validation
- Multimodal error troubleshooting guides

## 📊 Transaction Types & Journal Classification

The system supports these transaction types across all input formats and automatically classifies them into appropriate journal types:

### **💰 Invoice Transactions → PAYMENT Journal**
- **Trading**: Buy, Purchase, Subscription
- **Investment**: Placement, Investment, Acquisition
- **Cash Management**: Deposit, Transfer_Out, Payment

### **💸 Debit Memo Transactions → REFUND Journal**
- **Trading**: Sell, Disposal, Redemption
- **Income**: Dividend, Interest, Coupon, Revenue
- **Cash Management**: Withdrawal, Transfer_In, Receipt
- **Maturity**: Maturity, Repayment, Early Termination

### **🎯 Journal Mapping Logic**
Each transaction is automatically classified as either:
- **Invoice** → Creates **Payment** journal entries (money going out)
- **Debit Memo** → Creates **Refund** journal entries (money coming in)

## 🛠️ Development

### Project Structure

```
milestone-project-1-ai-agent/
├── src/                    # Main source code
│   ├── config.py          # Configuration settings
│   ├── services/          # API and external services
│   ├── processors/        # Document processing logic
│   │   ├── pdf_processor.py      # PDF processing
│   │   ├── text_processor.py     # Text/JSON/CSV processing
│   │   └── ai_processor.py       # Multimodal AI enhancement
│   └── ui/               # Streamlit UI components
├── examples/              # Sample input files
│   ├── sample_transaction_data.json
│   ├── sample_transaction_data.csv
│   └── sample_transaction_data.txt
├── main.py               # Main application entry point
├── app.py                # Alternative Streamlit entry point
├── pyproject.toml        # Project dependencies
├── .streamlit/           # Streamlit configuration
├── uploads/              # Temporary file storage
├── data/                 # Application data
└── docs/                 # Documentation
```

### Adding New File Formats

1. **Extend TextProcessor**: Add new format parsing in `src/processors/text_processor.py`
2. **Update Configuration**: Add file extension to `src/config.py`
3. **Enhance UI**: Update file upload component in `src/ui/components.py`
4. **Add Validation**: Extend file validator for new format

### Running in Development

```bash
# Enable debug mode
export DEBUG=true

# Run with auto-reload (option 1)
python main.py

# Run with auto-reload (option 2)
streamlit run main.py --server.runOnSave true
```

## 🔧 Troubleshooting

### Common Issues

1. **API Connection Failed** (PDF processing)
   - Verify API_BASE_URL is correct
   - Check if API token is valid and not expired
   - Ensure network connectivity to the backend

2. **File Upload Issues**
   - Check file size (must be < 50MB)
   - Ensure file format is supported (PDF, TXT, JSON, CSV)
   - Verify file encoding for text-based formats

3. **Processing Errors**
   - **PDF**: Check API backend logs and TReA service status
   - **Text/JSON/CSV**: Verify file format and structure
   - **JSON**: Validate JSON syntax and schema
   - **CSV**: Check column headers and encoding

4. **Format-Specific Issues**
   - **JSON**: Ensure `transactions` array exists in root object
   - **CSV**: Verify column headers match expected format
   - **Text**: Check line format matches pattern

### Debug Mode

Enable debug mode by setting `DEBUG=true` in your `.env` file for detailed error messages and stack traces.

## 📈 Sample Data

The `examples/` directory contains sample files for each supported format:

- `sample_transaction_data.json` - JSON format example
- `sample_transaction_data.csv` - CSV format example  
- `sample_transaction_data.txt` - Text format example

Use these files to test the multimodal capabilities of the system.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of the Milestone Project for the AI Agent course.

## 🆘 Support

For support and questions:

1. Check the troubleshooting section above
2. Review the system status page in the application
3. Test with sample files in the `examples/` directory
4. Contact the development team
5. Create an issue in the repository

---

**🎉 Now supports multimodal input processing! Built with ❤️ for automated treasury operations**


