import streamlit as st
import pandas as pd
import re
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Fungsi untuk membersihkan teks
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Menghapus mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # Menghapus hashtags
    text = re.sub(r'RT[\s]', '', text)  # Menghapus RT
    text = re.sub(r'http\S+', '', text)  # Menghapus link
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Menghapus karakter non-alfabet
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    text = text.strip()  # Menghapus spasi berlebih
    return text

# Fungsi untuk analisis sentimen dengan Vader
def sentiment_analysis(df):
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(word):
        score = analyzer.polarity_scores(word)
        if score['compound'] >= 0.05:
            return 'Positive'
        elif score['compound'] <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    # Tokenisasi teks
    df['tokens'] = df['cleaning'].apply(word_tokenize)
    
    # Analisis sentimen untuk setiap kata
    df['sentiment'] = df['tokens'].apply(lambda tokens: [get_sentiment(word) for word in tokens])

    st.write("Sentiment Analysis per Word:")
    st.write(df[['full_text', 'tokens', 'sentiment']])

    # Menghitung jumlah kategori sentimen
    sentiment_counts = pd.Series([sent for sublist in df['sentiment'] for sent in sublist]).value_counts()
    st.bar_chart(sentiment_counts)

# Fungsi untuk menampilkan WordCloud
def display_wordcloud(df, column):
    all_words = ' '.join(df[column].tolist())

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(all_words)
    filtered_words = [w for w in word_tokens if not w in stop_words and w not in string.punctuation]

    # Tampilkan WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Fungsi untuk menampilkan TopWords
def display_topwords(df, column):
    all_words = ' '.join(df[column].tolist())

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(all_words)
    filtered_words = [w for w in word_tokens if not w in stop_words and w not in string.punctuation]

    word_freq = pd.Series(filtered_words).value_counts().head(10)

    # Tampilkan 10 kata terbanyak dalam grafik batang
    plt.figure(figsize=(10, 5))
    sns.barplot(x=word_freq.values, y=word_freq.index, palette="viridis")
    plt.title('Top 10 Kata Paling Sering Muncul')
    plt.xlabel('Frekuensi')
    st.pyplot(plt)

# Inisialisasi variabel global untuk menyimpan data di luar halaman NLP Dashboard
if 'df' not in st.session_state:
    st.session_state['df'] = None

# UI Aplikasi Streamlit dengan Sidebar
st.title('Twitter NLP Dashboard')
st.sidebar.title('Navigasi')

# Halaman yang tersedia
page_names_to_funcs = {
    "NLP Dashboard": None,
    "WordCloud": display_wordcloud,
    "TopWords": display_topwords,
    "Sentiment Analysis": sentiment_analysis
}

# Pilihan di sidebar
selected_page = st.sidebar.selectbox("Pilih halaman", page_names_to_funcs.keys())

# Upload file CSV di halaman utama NLP Dashboard
if selected_page == "NLP Dashboard":
    st.subheader('Upload CSV dan Tampilkan Data Hasil Crawling')
    uploaded_file = st.file_uploader("Upload file CSV yang berisi tweet", type="csv")
    
    if uploaded_file is not None:
        # Menyimpan data di session_state agar bisa diakses oleh halaman lain
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df

        st.write("Data Hasil Crawling:")
        st.dataframe(df)

        # Bersihkan teks
        df['cleaning'] = df['full_text'].apply(cleaningText)
        st.session_state['df'] = df  # Simpan data yang telah dibersihkan
        st.write("Data Setelah Dibersihkan:")
        st.dataframe(df[['full_text', 'cleaning']])

# Jika halaman selain NLP Dashboard, cek apakah data sudah ada
if selected_page != "NLP Dashboard":
    if st.session_state['df'] is None:
        st.write("Silakan upload file CSV di halaman NLP Dashboard terlebih dahulu.")
    else:
        df = st.session_state['df']
        
        if selected_page == "WordCloud":
            st.subheader('WordCloud')
            display_wordcloud(df, 'cleaning')

        if selected_page == "TopWords":
            st.subheader('Top 10 Kata Paling Sering Muncul')
            display_topwords(df, 'cleaning')

        if selected_page == "Sentiment Analysis":
            st.subheader('Analisis Sentimen per Kata')
            sentiment_analysis(df)
