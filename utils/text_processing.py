"""
Text Preparocessing Utilities for the FAQ Chatbot project.

The preprocessing pipline:
1. Convert text to lowercase
2. Remove punctuation and special characters
3. Tokenize the text 
4. Apply stemming using NLTK
"""

import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import wordpunct_tokenize

# Initialize the NLTK stemmer.
stemmer = PorterStemmer()

# Common English stop words.
# Keeping them locally avoids requiring an addditional NLTK download.

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "here", "hers", 
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "me", "my", "my", "of",
    "on", "or", "our", "ours", "she", "should", "so", "than", "that", "the", "their", "theirs",
    "them", "there", "these", "they", "this", "those", "to", "too", "was", "we", "were", "what",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your", "yours"
    }

def preprocess_text(text):
    """Clean and preprocess text for NLP matching.
    Args:
        text (str): Input text.
    Returns:
    str: Cleaned and stemmed text.        
   """
# Make sure the input is a string.
    if not isinstance(text, str):
        return ""

# Convert text to lowercase.    
    text = text.lower()

# Remove URLs.
    text = re.sub(r"http?://\S+|www\.\S+", " ", text)

# Tokenize using NLTK>
    token = wordpunct_tokenize(text)

# Remove stop words and short words.
    tokens = [
        token for token in tokens
        if token.isalpha()
        and token not in STOP_WORDS
        and len(token) > 1
    ]

# Apply stemming.
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

# Return the processed text.
    return " ".join(stemmed_tokens)