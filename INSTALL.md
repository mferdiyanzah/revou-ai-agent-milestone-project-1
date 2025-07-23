# 🚀 Simple Installation Guide

## Quick Install (Recommended)

**Skip the build system entirely and install dependencies directly:**

```bash
# 1. Install dependencies directly from requirements.txt
pip install -r requirements.txt

# 2. Configure environment
cp env.template .env
# Edit .env with your settings

# 3. Run the application
python app.py
```

## Alternative Methods

### Method 1: Using pip directly
```bash
pip install streamlit requests openai pandas psycopg2-binary pgvector python-dotenv pydantic pydantic-settings numpy plotly
```

### Method 2: Using uv (if you prefer)
```bash
# Install dependencies without building the package
uv pip install -r requirements.txt
```

## Running the Application

Once dependencies are installed, you can run the app in multiple ways:

```bash
# Method 1: Direct streamlit
streamlit run app.py

# Method 2: Using the runner script
python run.py

# Method 3: Direct python execution
python app.py
```

## Configuration

Create a `.env` file from the template:

```bash
cp env.template .env
```

**Minimum required in .env:**
```bash
API_BASE_URL=http://192.168.74.211:30477
API_TOKEN=your_token_here
```

**Optional enhancements:**
```bash
# For AI features
OPENAI_API_KEY=your_openai_key

# For vector database
POSTGRES_HOST=localhost
POSTGRES_DB=trea_vector_db
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
```

## Troubleshooting

If you get build errors:
1. ✅ Use `pip install -r requirements.txt` instead of `uv sync`
2. ✅ Run `python app.py` instead of installing as a package
3. ✅ Make sure you're in the project directory

The application doesn't need to be installed as a package - it runs directly! 🎉 