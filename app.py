import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# Load the processed data (you'll need to save and load it)
@st.cache_data
def load_data():
    # In a real scenario, load from file or database
    # For now, we'll create sample data
    sample_data = {
        'text': [
            'DBS Bank India has excellent customer service and quick loan approval',
            'DBS app crashes too often, very frustrating experience',
            'DBS net banking interface is user-friendly and secure',
            'DBS credit card has high interest rates, not recommended',
            'DBS mobile banking works smoothly, love the features',
            'Poor customer support from DBS Bank India branch',
            'DBS digibank is innovative and convenient for daily banking',
            'DBS loan process is too lengthy and complicated',
            'Great experience with DBS debit card internationally',
            'DBS bank charges are reasonable compared to other banks'
        ],
        'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'neutral'],
        'emotion': ['happy', 'angry', 'satisfied', 'disappointed', 'happy', 'angry', 'satisfied', 'disappointed', 'happy', 'neutral'],
        'source': ['News', 'Reddit', 'News', 'Reddit', 'News', 'Reddit', 'News', 'Reddit', 'News', 'Reddit'],
        'platform': ['Economic Times', 'r/india', 'Business Standard', 'r/bangalore', 'Financial Express', 'r/mumbai', 'Times of India', 'r/delhi', 'Hindu Business', 'r/india'],
        'search_term': ['DBS Bank India', 'DBS app', 'DBS net banking', 'DBS credit card', 'DBS mobile banking', 'DBS Bank India', 'DBS digibank', 'DBS loan', 'DBS debit card', 'DBS Bank India'],
        'polarity': [0.8, -0.7, 0.6, -0.5, 0.7, -0.6, 0.8, -0.4, 0.9, 0.1],
        'date': pd.date_range(start='2024-06-01', periods=10, freq='D')
    }
    return pd.DataFrame(sample_data)

def main():
    st.set_page_config(page_title="DBS Bank Sentiment Dashboard", layout="wide", page_icon="🏦")
    
    st.title("🏦 DBS Bank India - Sentiment Analysis Dashboard")
    st.markdown("*Analyzing third-party mentions across social media, news, and forums*")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("📊 Filters")
    
    # Date filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['date'].min().date(), df['date'].max().date()),
        min_value=df['date'].min().date(),
        max_value=df['date'].max().date()
    )
    
    # Source filter
    sources = st.sidebar.multiselect(
        "Select Sources",
        options=df['source'].unique(),
        default=df['source'].unique()
    )
    
    # Product filter
    products = st.sidebar.multiselect(
        "Select Products/Keywords",
        options=df['search_term'].unique(),
        default=df['search_term'].unique()
    )
    
    # Filter data
    if len(date_range) == 2:
        df_filtered = df[
            (df['date'].dt.date >= date_range[0]) & 
            (df['date'].dt.date <= date_range[1]) &
            (df['source'].isin(sources)) &
            (df['search_term'].isin(products))
        ]
    else:
        df_filtered = df[
            (df['source'].isin(sources)) &
            (df['search_term'].isin(products))
        ]
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_mentions = len(df_filtered)
        st.metric("📈 Total Mentions", total_mentions)
    
    with col2:
        positive_count = len(df_filtered[df_filtered['sentiment'] == 'positive'])
        positive_pct = (positive_count / total_mentions * 100) if total_mentions > 0 else 0
        st.metric("👍 Positive", f"{positive_count} ({positive_pct:.1f}%)")
    
    with col3:
        negative_count = len(df_filtered[df_filtered['sentiment'] == 'negative'])
        negative_pct = (negative_count / total_mentions * 100) if total_mentions > 0 else 0
        st.metric("👎 Negative", f"{negative_count} ({negative_pct:.1f}%)")
    
    with col4:
        neutral_count = len(df_filtered[df_filtered['sentiment'] == 'neutral'])
        neutral_pct = (neutral_count / total_mentions * 100) if total_mentions > 0 else 0
        st.metric("😐 Neutral", f"{neutral_count} ({neutral_pct:.1f}%)")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sentiment Distribution")
        if not df_filtered.empty:
            sentiment_counts = df_filtered['sentiment'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                color_discrete_map={
                    'positive': '#2E8B57',
                    'negative': '#DC143C',
                    'neutral': '#4682B4'
                }
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data available for selected filters")
    
    with col2:
        st.subheader("😊 Emotion Classification")
        if not df_filtered.empty:
            emotion_counts = df_filtered['emotion'].value_counts()
            fig_emotion = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_emotion.update_layout(height=400)
            st.plotly_chart(fig_emotion, use_container_width=True)
        else:
            st.info("No data available for selected filters")
    
    # Time series
    st.subheader("📈 Sentiment Trend Over Time")
    if not df_filtered.empty:
        daily_sentiment = df_filtered.groupby([df_filtered['date'].dt.date, 'sentiment']).size().reset_index(name='count')
        fig_time = px.line(
            daily_sentiment,
            x='date',
            y='count',
            color='sentiment',
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C',
                'neutral': '#4682B4'
            }
        )
        fig_time.update_layout(height=400)
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No data available for selected filters")
    
    # Product breakdown
    st.subheader("🏦 Sentiment by Product/Keyword")
    if not df_filtered.empty:
        product_sentiment = df_filtered.groupby(['search_term', 'sentiment']).size().reset_index(name='count')
        fig_product = px.bar(
            product_sentiment,
            x='search_term',
            y='count',
            color='sentiment',
            color_discrete_map={
                'positive': '#2E8B57',
                'negative': '#DC143C',
                'neutral': '#4682B4'
            }
        )
        fig_product.update_layout(height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig_product, use_container_width=True)
    else:
        st.info("No data available for selected filters")
    
    # Top posts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👍 Top 3 Positive Mentions")
        positive_posts = df_filtered[df_filtered['sentiment'] == 'positive'].nlargest(3, 'polarity')
        for idx, row in positive_posts.iterrows():
            with st.expander(f"💚 {row['platform']} - Score: {row['polarity']:.2f}"):
                st.write(f"**Text:** {row['text']}")
                st.write(f"**Source:** {row['source']} | **Platform:** {row['platform']}")
                st.write(f"**Date:** {row['date'].strftime('%Y-%m-%d')}")
    
    with col2:
        st.subheader("👎 Top 3 Negative Mentions")
        negative_posts = df_filtered[df_filtered['sentiment'] == 'negative'].nsmallest(3, 'polarity')
        for idx, row in negative_posts.iterrows():
            with st.expander(f"🔴 {row['platform']} - Score: {row['polarity']:.2f}"):
                st.write(f"**Text:** {row['text']}")
                st.write(f"**Source:** {row['source']} | **Platform:** {row['platform']}")
                st.write(f"**Date:** {row['date'].strftime('%Y-%m-%d')}")
    
    # Word Cloud
    st.subheader("☁️ Word Cloud - Key Terms")
    if not df_filtered.empty:
        all_text = ' '.join(df_filtered['text'].astype(str))
        
        # Remove common words
        stop_words = {'dbs', 'bank', 'india', 'banking', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        try:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                stopwords=stop_words
            ).generate(all_text)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        except:
            st.info("Unable to generate word cloud with current data")
    
    # Data export
    st.subheader("📥 Export Data")
    if st.button("Download Data as CSV"):
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV File",
            data=csv,
            file_name=f"dbs_sentiment_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard created for educational purposes | Data refreshed daily*")

if __name__ == "__main__":
    main()
