import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from wordcloud import WordCloud
from collections import Counter
import math

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

def plot_word_length_distribution(df):
    """
    Create a histogram showing the distribution of word counts in messages
    """
    fig = px.histogram(
        df,
        x='word_count',
        color='label',
        nbins=15,
        title='Word Count Distribution',
        color_discrete_map={'Spam': '#D83B01', 'Not Spam': '#107C41'},
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title='Number of Words',
        yaxis_title='Count of Messages',
        bargap=0.1,
        height=400
    )
    
    return fig

def plot_treemap_word_frequency(freq_data):
    """
    Create a treemap visualization of word frequencies
    """
    # Merge the two dataframes
    spam_df = freq_data['spam'].copy()
    non_spam_df = freq_data['non_spam'].copy()
    
    # Add category labels
    spam_df['category'] = 'Spam'
    non_spam_df['category'] = 'Not Spam'
    
    # Combine the dataframes
    combined_df = pd.concat([spam_df, non_spam_df])
    
    # Get top words for visualization
    top_words = combined_df.groupby('word')['frequency'].sum().sort_values(ascending=False).head(30).index.tolist()
    filtered_df = combined_df[combined_df['word'].isin(top_words)]
    
    # Create treemap
    fig = px.treemap(
        filtered_df,
        path=['category', 'word'],
        values='frequency',
        color='category',
        color_discrete_map={'Spam': '#D83B01', 'Not Spam': '#107C41'},
        title='Word Frequency Treemap'
    )
    
    fig.update_layout(
        height=500
    )
    
    return fig

def plot_radar_chart(df):
    """
    Create a radar chart comparing spam vs non-spam message characteristics
    """
    # Extract metrics
    spam_df = df[df['label'] == 'Spam']
    non_spam_df = df[df['label'] == 'Not Spam']
    
    # Calculate various metrics
    spam_metrics = {
        'avg_length': spam_df['message_length'].mean(),
        'avg_words': spam_df['word_count'].mean(),
        'max_length': min(spam_df['message_length'].max(), 1000),  # Cap for visualization
        'special_chars': spam_df['message'].apply(lambda x: sum(c in '!@#$%^&*()' for c in x)).mean()
    }
    
    non_spam_metrics = {
        'avg_length': non_spam_df['message_length'].mean(),
        'avg_words': non_spam_df['word_count'].mean(),
        'max_length': min(non_spam_df['message_length'].max(), 1000),  # Cap for visualization
        'special_chars': non_spam_df['message'].apply(lambda x: sum(c in '!@#$%^&*()' for c in x)).mean()
    }
    
    # Normalize values between 0 and 1 for radar chart
    all_values = list(spam_metrics.values()) + list(non_spam_metrics.values())
    max_values = {k: max(spam_metrics[k], non_spam_metrics[k]) for k in spam_metrics}
    
    # Normalize metrics
    spam_metrics_norm = {k: v/max_values[k] for k, v in spam_metrics.items()}
    non_spam_metrics_norm = {k: v/max_values[k] for k, v in non_spam_metrics.items()}
    
    # Prepare data for radar chart
    categories = ['Average Length', 'Average Word Count', 'Maximum Length', 'Special Characters']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[spam_metrics_norm['avg_length'], spam_metrics_norm['avg_words'], 
           spam_metrics_norm['max_length'], spam_metrics_norm['special_chars']],
        theta=categories,
        fill='toself',
        name='Spam',
        line_color='#D83B01'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[non_spam_metrics_norm['avg_length'], non_spam_metrics_norm['avg_words'], 
           non_spam_metrics_norm['max_length'], non_spam_metrics_norm['special_chars']],
        theta=categories,
        fill='toself',
        name='Not Spam',
        line_color='#107C41'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Message Characteristics Comparison",
        height=500
    )
    
    return fig

def plot_sunburst_chart(df):
    """
    Create a sunburst chart showing message distribution by length and category
    """
    # Create length categories
    bins = [0, 50, 100, 150, 200, 300, 500, 1000]
    labels = ['Very Short', 'Short', 'Medium-Short', 'Medium', 'Medium-Long', 'Long', 'Very Long']
    
    # Add length category
    df['length_category'] = pd.cut(df['message_length'], bins=bins, labels=labels)
    
    # Aggregate data
    sunburst_data = df.groupby(['label', 'length_category']).size().reset_index(name='count')
    
    # Create sunburst chart
    fig = px.sunburst(
        sunburst_data,
        path=['label', 'length_category'],
        values='count',
        color='label',
        color_discrete_map={'Spam': '#D83B01', 'Not Spam': '#107C41'},
        title='Message Distribution by Category and Length'
    )
    
    fig.update_layout(
        height=550
    )
    
    return fig

def plot_message_heatmap(df):
    """
    Create a heatmap visualization of message characteristics by time
    """
    # Create a simple proxy for time (since we don't have timestamps)
    # We'll use row index as a proxy for time
    df = df.copy()
    df['time_bin'] = pd.qcut(np.arange(len(df)), 10, labels=False)
    
    # Create a pivot table for the heatmap
    heatmap_data = df.pivot_table(
        index='time_bin',
        columns='label',
        values='message_length',
        aggfunc='mean'
    ).fillna(0)
    
    # Create annotation text
    annotations = go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=[f'Batch {i+1}' for i in range(len(heatmap_data))],
        colorscale='RdYlGn',
        showscale=True,
        text=[[f'{val:.1f}' for val in row] for row in heatmap_data.values],
        texttemplate="%{text}",
        colorbar=dict(title='Avg. Length')
    )
    
    # Create the figure
    fig = go.Figure(data=annotations)
    
    fig.update_layout(
        title='Average Message Length by Category and Time Batch',
        xaxis_title='Message Category',
        yaxis_title='Time Batch',
        height=400
    )
    
    return fig
