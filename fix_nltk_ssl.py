#!/usr/bin/env python3
"""Fix NLTK SSL certificate issues on macOS"""

import ssl
import nltk
import os

# Create unverified SSL context
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data path
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download required NLTK data
print("Downloading NLTK data packages...")
try:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    nltk.download('wordnet', download_dir=nltk_data_dir)
    print("✅ NLTK data downloaded successfully!")
except Exception as e:
    print(f"❌ Error downloading NLTK data: {e}")

# Verify installation
print("\nVerifying NLTK data...")
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    test_text = "This is a test sentence."
    tokens = word_tokenize(test_text)
    stop_words = stopwords.words('english')
    print(f"✅ NLTK is working! Tokenized: {tokens[:3]}...")
    print(f"✅ Stopwords loaded: {len(stop_words)} words")
except Exception as e:
    print(f"❌ NLTK verification failed: {e}")