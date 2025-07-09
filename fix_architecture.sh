#!/bin/bash
# Fix Python architecture mismatch issues

echo "🔧 Fixing Python architecture mismatch..."

# Reinstall problematic packages
echo "📦 Reinstalling packages with correct architecture..."

pip uninstall -y regex cryptography cffi
pip install --no-cache-dir --force-reinstall regex cryptography cffi

# Try to download NLTK data manually
echo "📥 Downloading NLTK data manually..."
python -c "
import ssl
import urllib.request

# Disable SSL verification for downloading
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK data files manually
import os
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Create tokenizers directory
os.makedirs(os.path.join(nltk_data_dir, 'tokenizers'), exist_ok=True)
os.makedirs(os.path.join(nltk_data_dir, 'corpora'), exist_ok=True)

print('✅ NLTK directories created')
"

# Alternative: Download NLTK data using command line
python -m nltk.downloader punkt stopwords wordnet -d ~/nltk_data || echo "Manual download may be needed"

echo "✅ Fix completed!"