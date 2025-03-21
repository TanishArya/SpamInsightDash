import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

def plot_spam_distribution(df):
    """
    Create a pie chart showing spam vs. non-spam distribution
    """
    count_data = df['label'].value_counts().reset_index()
    count_data.columns = ['Category', 'Count']
    
    fig = px.pie(
        count_data, 
        values='Count', 
        names='Category',
        title='Spam vs. Non-Spam Distribution',
        color_discrete_map={'Spam': '#D83B01', 'Not Spam': '#107C41'},
        hole=0.4
    )
    
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        marker=dict(line=dict(color='#FFFFFF', width=1))
    )
    
    fig.update_layout(
        legend_title_text='Message Type',
        height=400
    )
    
    return fig

def plot_word_frequency(freq_data):
    """
    Create a bar chart comparing word frequencies in spam vs. non-spam messages
    """
    # Merge the two dataframes
    spam_df = freq_data['spam']
    non_spam_df = freq_data['non_spam']
    
    # Get all unique words
    all_words = set(spam_df['word']).union(set(non_spam_df['word']))
    
    # Create a merged dataframe
    merged_data = []
    
    for word in all_words:
        spam_freq = spam_df[spam_df['word'] == word]['frequency'].values[0] if word in spam_df['word'].values else 0
        non_spam_freq = non_spam_df[non_spam_df['word'] == word]['frequency'].values[0] if word in non_spam_df['word'].values else 0
        
        # Only include words that appear in either category's top words
        if spam_freq > 0 or non_spam_freq > 0:
            merged_data.append({
                'word': word,
                'Spam': spam_freq,
                'Not Spam': non_spam_freq
            })
    
    merged_df = pd.DataFrame(merged_data)
    
    # Sort by total frequency
    merged_df['total'] = merged_df['Spam'] + merged_df['Not Spam']
    merged_df = merged_df.sort_values('total', ascending=False).head(15)
    
    # Melt the dataframe for Plotly
    melted_df = pd.melt(
        merged_df,
        id_vars=['word'],
        value_vars=['Spam', 'Not Spam'],
        var_name='Category',
        value_name='Frequency'
    )
    
    fig = px.bar(
        melted_df,
        x='word',
        y='Frequency',
        color='Category',
        title='Word Frequency Comparison: Spam vs. Non-Spam',
        barmode='group',
        color_discrete_map={'Spam': '#D83B01', 'Not Spam': '#107C41'}
    )
    
    fig.update_layout(
        xaxis_title='Word',
        yaxis_title='Frequency',
        xaxis={'categoryorder':'total descending'},
        height=500
    )
    
    return fig

def plot_message_length_comparison(df):
    """
    Create a box plot comparing message lengths of spam vs. non-spam
    """
    fig = px.box(
        df,
        x='label',
        y='message_length',
        color='label',
        title='Message Length Comparison',
        color_discrete_map={'Spam': '#D83B01', 'Not Spam': '#107C41'}
    )
    
    fig.update_layout(
        xaxis_title='Message Type',
        yaxis_title='Message Length (characters)',
        height=400
    )
    
    return fig

def plot_character_count_distribution(df):
    """
    Create a histogram showing distribution of message lengths
    """
    fig = px.histogram(
        df,
        x='message_length',
        color='label',
        nbins=20,
        title='Character Count Distribution',
        color_discrete_map={'Spam': '#D83B01', 'Not Spam': '#107C41'},
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title='Message Length (characters)',
        yaxis_title='Count',
        bargap=0.1,
        height=400
    )
    
    return fig

def plot_word_clouds(df, message_type='spam'):
    """
    Create word clouds for spam and non-spam messages
    """
    # Filter by message type
    if message_type.lower() == 'spam':
        filtered_df = df[df['label'] == 'Spam']
        color = 'Oranges'
    else:
        filtered_df = df[df['label'] == 'Not Spam']
        color = 'Greens'
    
    # Combine all messages
    text = ' '.join(filtered_df['clean_message'].tolist())
    
    # Create the word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap=color,
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

def plot_confusion_matrix(df):
    """
    Create a heatmap visualization of the confusion matrix
    """
    # Calculate confusion matrix values
    tp = len(df[(df['is_spam'] == 1) & (df['predicted_spam'] == 1)])
    fp = len(df[(df['is_spam'] == 0) & (df['predicted_spam'] == 1)])
    fn = len(df[(df['is_spam'] == 1) & (df['predicted_spam'] == 0)])
    tn = len(df[(df['is_spam'] == 0) & (df['predicted_spam'] == 0)])
    
    # Create the matrix
    z = [[tp, fp], [fn, tn]]
    
    x = ['Predicted Spam', 'Predicted Not Spam']
    y = ['Actual Spam', 'Actual Not Spam']
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=[[0, '#107C41'], [1, '#D83B01']],
        hoverongaps=False,
        showscale=False,
        text=[[f'True Positive: {tp}', f'False Positive: {fp}'], 
              [f'False Negative: {fn}', f'True Negative: {tn}']],
        texttemplate="%{text}",
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        height=400
    )
    
    return fig
