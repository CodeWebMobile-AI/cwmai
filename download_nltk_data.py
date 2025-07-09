#!/usr/bin/env python3
"""Download NLTK data manually to bypass SSL issues"""

import os
import urllib.request
import zipfile
import tarfile
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# NLTK data directory
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# NLTK data URLs
nltk_data_urls = {
    'punkt': {
        'url': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip',
        'path': 'tokenizers'
    },
    'stopwords': {
        'url': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip',
        'path': 'corpora'
    },
    'wordnet': {
        'url': 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip',
        'path': 'corpora'
    }
}

def download_and_extract(name, info):
    """Download and extract NLTK data package"""
    try:
        # Create target directory
        target_dir = os.path.join(nltk_data_dir, info['path'])
        os.makedirs(target_dir, exist_ok=True)
        
        # Download file
        print(f"📥 Downloading {name}...")
        temp_file = os.path.join(nltk_data_dir, f'{name}.zip')
        urllib.request.urlretrieve(info['url'], temp_file)
        
        # Extract file
        print(f"📦 Extracting {name}...")
        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        
        # Clean up
        os.remove(temp_file)
        print(f"✅ {name} installed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error installing {name}: {e}")
        return False

# Download all packages
print("🚀 Starting NLTK data download...")
success_count = 0
for name, info in nltk_data_urls.items():
    if download_and_extract(name, info):
        success_count += 1

print(f"\n📊 Downloaded {success_count}/{len(nltk_data_urls)} packages")

# Test NLTK
print("\n🧪 Testing NLTK...")
try:
    import nltk
    nltk.data.path.append(nltk_data_dir)
    
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    
    test_text = "This is a test sentence."
    tokens = word_tokenize(test_text)
    stop_words = stopwords.words('english')
    
    print(f"✅ NLTK is working!")
    print(f"   Tokens: {tokens}")
    print(f"   Stopwords: {len(stop_words)} words loaded")
    
except Exception as e:
    print(f"❌ NLTK test failed: {e}")
    print("\nTry setting NLTK_DATA environment variable:")
    print(f"export NLTK_DATA={nltk_data_dir}")