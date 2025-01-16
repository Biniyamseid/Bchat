# Developer Guide

## Environment Setup

### Virtual Environment Issues

If you encounter package installation issues or see "requirement already satisfied" errors pointing to system Python paths, follow these steps:

```bash
# 1. Deactivate current virtual environment (if any)
deactivate

# 2. Remove existing virtual environment
rm -rf venv

# 3. Create new virtual environment
python3 -m venv venv

# 4. Activate new virtual environment
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows

# 5. Upgrade pip
pip install --upgrade pip

# 6. Install requirements
pip install -r requirements.txt
```

### Verifying Correct Installation

To verify packages are installed in the virtual environment:
```bash
# Should show path in your venv directory
python -c "import pandas; print(pandas.__file__)"
```

### Common Issues

1. **System-wide Installation**: If packages are being installed in `/usr/local/lib/` instead of your virtual environment, follow the reset steps above.
2. **Permission Issues**: Never use `sudo pip install` in a virtual environment.
3. **Package Conflicts**: If you see version conflicts, try installing major dependencies first:
   ```bash
   pip install pandas
   pip install langchain-openai langchain-community
   pip install -r requirements.txt
   ```

## Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the development server:
```bash
uvicorn app.main:app --reload
```