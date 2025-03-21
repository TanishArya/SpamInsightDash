import pandas as pd
import numpy as np
import os
import re
from collections import Counter

def load_sample_data():
    """
    Load the sample spam detection data
    """
    try:
        # Check if the file exists in the attached_assets folder
        if os.path.exists('attached_assets/spam.csv'):
            df = pd.read_csv('attached_assets/spam.csv', encoding='latin1')
            # The first two columns are likely label and message
            df = df.iloc[:, :2]
            df.columns = ['label', 'message']
            return df
        else:
            # Fallback if file doesn't exist at expected location
            raise FileNotFoundError("Sample data file not found")
    except Exception as e:
        raise Exception(f"Error loading sample data: {str(e)}")

def preprocess_data(df):
    """
    Preprocess the data for analysis
    """
    # Ensure we have the expected columns
    if len(df.columns) < 2:
        raise ValueError("Data must have at least two columns (label and message)")
    
    # Rename columns if needed
    if df.columns[0].lower() not in ['label', 'class', 'spam', 'category', 'type', 'v1']:
        raise ValueError("First column should contain spam/not spam labels")
    
    if df.columns[1].lower() not in ['message', 'text', 'sms', 'content', 'v2']:
        raise ValueError("Second column should contain message text")
    
    # Create a new dataframe with standardized column names
    processed_df = pd.DataFrame()
    processed_df['label'] = df.iloc[:, 0]
    processed_df['message'] = df.iloc[:, 1]
    
    # Standardize labels
    label_mapping = {
        'ham': 'Not Spam',
        'spam': 'Spam',
        '0': 'Not Spam',
        '1': 'Spam',
        0: 'Not Spam',
        1: 'Spam'
    }
    
    # Convert labels to standard format
    processed_df['label'] = processed_df['label'].apply(
        lambda x: label_mapping.get(x, x)
    )
    
    # If labels are still not standardized, try to infer them
    if not all(processed_df['label'].isin(['Spam', 'Not Spam'])):
        unique_labels = processed_df['label'].unique()
        if len(unique_labels) == 2:
            # Assume the less frequent label is spam
            counts = processed_df['label'].value_counts()
            spam_label = counts.index[counts.argmin()]
            not_spam_label = counts.index[counts.argmax()]
            
            processed_df['label'] = processed_df['label'].replace({
                spam_label: 'Spam',
                not_spam_label: 'Not Spam'
            })
    
    # Calculate message length
    processed_df['message_length'] = processed_df['message'].apply(len)
    
    # Count words in each message
    processed_df['word_count'] = processed_df['message'].apply(lambda x: len(str(x).split()))
    
    # Clean text for analysis
    processed_df['clean_message'] = processed_df['message'].apply(clean_text)
    
    # Create binary labels for confusion matrix
    processed_df['is_spam'] = (processed_df['label'] == 'Spam').astype(int)
    
    # Simulate predicted values based on a basic rule
    # In a real system, this would come from an actual classifier
    processed_df['predicted_spam'] = processed_df['message'].apply(
        lambda x: 1 if any(word in x.lower() for word in ['free', 'win', 'prize', 'cash', 'claim']) else 0
    )
    
    return processed_df

def clean_text(text):
    """
    Clean text for analysis by removing punctuation, special characters, etc.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_summary_stats(df):
    """
    Calculate summary statistics for the dashboard
    """
    total_messages = len(df)
    spam_count = len(df[df['label'] == 'Spam'])
    non_spam_count = len(df[df['label'] == 'Not Spam'])
    
    spam_percentage = (spam_count / total_messages) * 100 if total_messages > 0 else 0
    non_spam_percentage = (non_spam_count / total_messages) * 100 if total_messages > 0 else 0
    
    avg_message_length = df['message_length'].mean()
    avg_spam_length = df[df['label'] == 'Spam']['message_length'].mean()
    avg_non_spam_length = df[df['label'] == 'Not Spam']['message_length'].mean()
    
    return {
        'total_messages': total_messages,
        'spam_count': spam_count,
        'non_spam_count': non_spam_count,
        'spam_percentage': spam_percentage,
        'non_spam_percentage': non_spam_percentage,
        'avg_message_length': avg_message_length,
        'avg_spam_length': avg_spam_length,
        'avg_non_spam_length': avg_non_spam_length
    }
