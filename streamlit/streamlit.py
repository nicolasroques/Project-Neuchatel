import streamlit as st
import pandas as pd
import joblib
import spacy
from gtts import gTTS
from io import BytesIO
from deep_translator import GoogleTranslator
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
from textstat import lexicon_count, sentence_count, flesch_reading_ease
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import requests
from tempfile import NamedTemporaryFile


# Download the set of stop words the first time
nltk.download('stopwords')

# Function Definitions
def play_audio(sentence):
    tts = gTTS(sentence, lang='fr')
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    st.audio(audio_data, format='audio/mp3')

def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/nicolasroques/Project-Neuchatel/main/streamlit/files/training_data.csv')

def get_pos_tags(sentence):
    doc = nlp(sentence)
    return [(token.text, token.pos_) for token in doc]

def create_download_link(df, filename, text="Download CSV file"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'

def translate_word(word):
    try:
        return translator.translate(word)
    except Exception as e:
        return word

def display_pos_tags_with_translation(pos_tags):
    pos_tags_with_translation = []
    for word, pos in pos_tags:
        translated_word = translate_word(word)
        pos_tags_with_translation.append((word, pos, translated_word))
    pos_df = pd.DataFrame(pos_tags_with_translation, columns=['Word', 'POS Tag', 'Translation'])
    st.table(pos_df)

def average_word_length(text):
    words = text.split()
    return sum(len(word) for word in words) / len(words)

def average_sentence_length(text):
    num_sentences = sentence_count(text)
    num_words = lexicon_count(text, removepunct=True)
    return num_words / num_sentences if num_sentences > 0 else 0

def readability_score(text):
    return flesch_reading_ease(text)

def rare_word_analysis(text, common_words_set):
    words = set(text.split())
    rare_words = words - common_words_set
    return len(rare_words), rare_words

def download_and_load_model(url):
    response = requests.get(url)
    with NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(response.content)
        return joblib.load(tmp_file.name)
def load_common_words(url):
    response = requests.get(url)
    return set(response.text.splitlines())

def extract_and_count_pos_tags(data):
    pos_counts = Counter()
    for sentence in data['sentence']:
        doc = nlp(sentence)
        pos_counts.update([token.pos_ for token in doc])
    return pos_counts

def plot_pos_distribution(pos_counts):
    pos_df = pd.DataFrame.from_dict(pos_counts, orient='index').reset_index()
    pos_df.columns = ['POS Tag', 'Count']
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Count', y='POS Tag', data=pos_df.sort_values(by='Count', ascending=False))
    plt.title('POS Tag Distribution in Sentences')
    plt.xlabel('Count')
    plt.ylabel('POS Tag')
    st.pyplot(plt)

# Streamlit Configuration
st.set_page_config(page_title="Analyzing French Text Difficulty", page_icon="📖",)

# Load Data and Models
data = load_data()
nlp = spacy.load("fr_core_news_sm")
model_url = 'https://raw.githubusercontent.com/nicolasroques/Project-Neuchatel/main/streamlit/files/text_classifier_logistic_regression.pkl'
model = download_and_load_model(model_url)

translator = GoogleTranslator(source='auto', target='en')
common_words_url = 'https://raw.githubusercontent.com/nicolasroques/Project-Neuchatel/main/streamlit/files/common_words.txt'
common_words_set = load_common_words(common_words_url)
stop_words = set(stopwords.words('french'))

# Streamlit User Interface
st.title('Analyzing French Text Difficulty')

# Sidebar Options
st.sidebar.header('Filter Options')
difficulty = st.sidebar.multiselect('Select Difficulty Level:', options=data['difficulty'].unique(), default=data['difficulty'].unique())
search_query = st.sidebar.text_input("Search for words in sentences:")
sort_options = ['Alphabetical', 'Sentence Length']
sort_by = st.sidebar.selectbox('Sort Sentences By:', sort_options)

# Data Filtering and Sorting
filtered_data = data[data['difficulty'].isin(difficulty)]
if search_query:
    filtered_data = filtered_data[filtered_data['sentence'].str.contains(search_query, case=False, na=False)]
if sort_by == 'Alphabetical':
    filtered_data = filtered_data.sort_values(by='sentence')
elif sort_by == 'Sentence Length':
    filtered_data['length'] = filtered_data['sentence'].apply(len)
    filtered_data = filtered_data.sort_values(by='length')

# Prediction Interface
st.header("Prediction of difficulty level")
sentence = st.text_input('Enter a sentence to predict its difficulty level.')
if st.button('Predict'):
    prediction = model.predict([sentence])
    st.write('Predicted Difficulty Level: ', prediction[0])

# Complexity Analysis Interface
st.header("Complexity analysis of sentences")
text = st.text_input("Enter a sentence for a complexity analysis:")
if st.button('Generate complexity analysis'):
    # Complexity calculations
    avg_word_length = average_word_length(text)
    avg_sent_length = average_sentence_length(text)
    readability = readability_score(text)
    rare_words_count, rare_words = rare_word_analysis(text, common_words_set)

    # Display results
    st.write(f"Average Word Length: {avg_word_length}")
    st.write(f"Average Sentence Length: {avg_sent_length}")
    st.write(f"Readability Score: {readability}")
    st.write(f"Number of Rare Words: {rare_words_count}")
    st.write("Rare Words:", ', '.join(rare_words))

    # Visualizations
    fig, ax = plt.subplots()
    ax.bar(['Avg Word Length', 'Avg Sentence Length', 'Readability Score'], [avg_word_length, avg_sent_length, readability])
    st.pyplot(fig)

# Further Analysis and Visualizations
st.header("Further Analysis")
# Your further analysis and visualizations here...

# User Interaction with Filtered Data
st.subheader('Interacting with Filtered Sentences')

# Initialize the session state variable if it's not already set
if 'show_all' not in st.session_state:
    st.session_state.show_all = False

# Toggle button for showing all sentences
if st.button('Show All Sentences' if not st.session_state.show_all else 'Show Only First 10 Sentences'):
    st.session_state.show_all = not st.session_state.show_all

# Counter for displayed sentences
displayed_sentences = 0

for index, row in filtered_data.iterrows():
    # Check if we should display this sentence
    if displayed_sentences < 10 or st.session_state.show_all:
        sentence = row['sentence']
        difficulty_level = row['difficulty']
        sentence_with_difficulty = f"{sentence} (Difficulty: {difficulty_level})"
        if st.button(sentence_with_difficulty, key=row['id']):
            # Display POS Tags, audio, and translation
            pos_tags = get_pos_tags(sentence)
            display_pos_tags_with_translation(pos_tags)
            play_audio(sentence)
            translation = translator.translate(sentence)
            st.write(f"English Translation: {translation}")

        displayed_sentences += 1
    else:
        break


# Data Export Option
if st.button('Export Filtered Sentences'):
    # Exporting logic here...
    filtered_data_csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.success('Filtered sentences exported to CSV file.')
    st.download_button(
        label="Download Filtered Sentences as CSV",
        data=filtered_data_csv,
        file_name='filtered_sentences.csv',
        mime='text/csv',
    )

# Visualization of Sentence Length by Difficulty Level
st.divider()
st.header("Analysis of Sentence Length by Difficulty Level")
data['sentence_length'] = data['sentence'].apply(len)
avg_length_by_difficulty = data.groupby('difficulty')['sentence_length'].mean()

plt.figure(figsize=(10, 6))
sns.barplot(x=avg_length_by_difficulty.index, y=avg_length_by_difficulty.values)
plt.xlabel('Difficulty Level')
plt.ylabel('Average Sentence Length')
plt.title('Average Sentence Length by Difficulty Level')
st.pyplot(plt)

# POS Tag Distribution Analysis
st.divider()
st.header("POS Tag Distribution Analysis")
pos_counts = extract_and_count_pos_tags(data)
plot_pos_distribution(pos_counts)

# Footer
st.divider()
st.write("Streamlit App for Analyzing French Text Difficulty")
