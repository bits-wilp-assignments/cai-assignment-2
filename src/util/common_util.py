import nltk

# Ensure the necessary NLTK resources are available
def ensure_nltk_resources():
    try:
        # Check if 'punkt' is already available
        nltk.data.find('tokenizers/punkt')
        print("NLTK 'punkt' is already installed.")
    except LookupError:
        # If not found, download it
        print("NLTK 'punkt' not found. Downloading...")
        nltk.download('punkt')