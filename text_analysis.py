import pandas as pd
from collections import Counter
import re
import string
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def extract_word_frequencies(df):
    """
    Extract word frequencies from messages, separated by spam and not spam
    """
    # Separate spam and non-spam messages
    spam_messages = df[df['label'] == 'Spam']['clean_message']
    non_spam_messages = df[df['label'] == 'Not Spam']['clean_message']
    
    # Get stop words
    stop_words = set(stopwords.words('english'))
    
    # Process words for spam messages
    spam_words = []
    for message in spam_messages:
        words = message.lower().split()
        # Filter out stop words and short words
        words = [word for word in words if word not in stop_words and len(word) > 2]
        spam_words.extend(words)
    
    # Process words for non-spam messages
    non_spam_words = []
    for message in non_spam_messages:
        words = message.lower().split()
        # Filter out stop words and short words
        words = [word for word in words if word not in stop_words and len(word) > 2]
        non_spam_words.extend(words)
    
    # Count frequencies
    spam_word_freq = Counter(spam_words)
    non_spam_word_freq = Counter(non_spam_words)
    
    # Get the top 20 words for each category
    top_spam_words = spam_word_freq.most_common(20)
    top_non_spam_words = non_spam_word_freq.most_common(20)
    
    # Convert to DataFrames
    spam_df = pd.DataFrame(top_spam_words, columns=['word', 'frequency'])
    non_spam_df = pd.DataFrame(top_non_spam_words, columns=['word', 'frequency'])
    
    return {
        'spam': spam_df,
        'non_spam': non_spam_df,
        'spam_raw': spam_word_freq,
        'non_spam_raw': non_spam_word_freq
    }

def get_message_length_stats(df):
    """
    Calculate statistics about message lengths
    """
    # Group by label and calculate length statistics
    length_stats = df.groupby('label')['message_length'].agg(['min', 'max', 'mean', 'median', 'std']).reset_index()
    
    # Create bins for message length
    bins = [0, 50, 100, 150, 200, 300, 500, 1000, float('inf')]
    labels = ['0-50', '51-100', '101-150', '151-200', '201-300', '301-500', '501-1000', '1000+']
    
    # Add length bins
    df['length_bin'] = pd.cut(df['message_length'], bins=bins, labels=labels)
    
    # Count messages in each bin by spam status
    length_distribution = df.groupby(['label', 'length_bin']).size().unstack(fill_value=0)
    
    return {
        'stats': length_stats,
        'distribution': length_distribution
    }

def analyze_common_words(df):
    """
    Analyze and return the most common words in spam and non-spam messages
    """
    word_frequencies = extract_word_frequencies(df)
    
    # Create DataFrames for display
    spam_words_df = pd.DataFrame(word_frequencies['spam_raw'].most_common(10), 
                                columns=['Word', 'Frequency'])
    
    non_spam_words_df = pd.DataFrame(word_frequencies['non_spam_raw'].most_common(10), 
                                   columns=['Word', 'Frequency'])
    
    return {
        'spam_words': spam_words_df,
        'non_spam_words': non_spam_words_df
    }
