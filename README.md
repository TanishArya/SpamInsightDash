# Spam Detection Visualization Dashboard

A comprehensive web-based dashboard for analyzing spam detection patterns and visualizing key metrics, built with Streamlit and Python data visualization libraries.

## Features

- **Interactive Visualizations**: Explore spam detection data through multiple visualization types
- **Multi-dimensional Analysis**: Analyze text content, message length, word frequency, and more
- **Professional Design**: Built following Microsoft Power BI and Google Analytics design principles
- **Responsive Layout**: Adapts to different screen sizes with a clean, card-based interface
- **Export Functionality**: Download data in various formats for further analysis

## Visualization Categories

1. **Classification Metrics**: Distribution charts and confusion matrix visualization
2. **Content Analysis**: Word clouds and common words analysis
3. **Word Frequency**: Comparison of word usage patterns between spam and non-spam
4. **Length Analysis**: Message length comparisons and distribution visualizations
5. **Advanced Visualizations**: Complex charts including heatmaps and scatter plots
6. **Comparison Charts**: Direct multi-dimensional comparisons via radar charts and other visualizations

## Project Structure

- **app.py**: Main application file with the dashboard interface
- **visualization.py**: Visualization functions for various chart types
- **text_analysis.py**: Text analysis utilities for processing message content
- **util.py**: Utility functions for data loading and preprocessing
- **dependencies.txt**: Documentation of package versions used

## Setup and Running

### Dependencies

The project uses multiple Python libraries as documented in `dependencies.txt`:

```
streamlit==1.30.0
pandas==2.1.4
numpy==1.26.3
matplotlib==3.8.3
plotly==5.18.0
wordcloud==1.9.3
nltk==3.8.1
```

### Running the Application

To run the dashboard:

```bash
streamlit run app.py --server.port 5000
```

## Color Scheme

- **Primary**: #0078D4 (Microsoft blue)
- **Secondary**: #107C41 (Green)
- **Accent**: #D83B01 (Warning orange)
- **Background**: #F5F5F5 (Light gray)
- **Text**: #252525 (Dark gray)

## Data Format

The application expects a CSV file with at least:
- A column for message content
- A column for spam/not spam classification

Sample data is included for demonstration purposes.

## License

This project is available for educational and professional use.