############### Text Analytics Group Project #############################


# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.util import ngrams
import string
import warnings

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# STEP 1: Data Preparation
# ========================

# Define the path to your files
file_path_template = "C:/Users/aadel/OneDrive/Documents/Mcgill Masters/3. Winter Term/Winter 1/Text Analytics/Group Project/Data/NLP_Papers_{}.csv"

# Load and combine the datasets for years 2016 to 2020
df_list = [pd.read_csv(file_path_template.format(year)) for year in range(2016, 2021)]
df_combined = pd.concat(df_list, ignore_index=True)

# Calculate the overall median citation count and label papers
overall_median_citations = df_combined['Cites'].median()
df_combined['Citation_Class'] = df_combined['Cites'].apply(lambda x: 1 if x > overall_median_citations else 0)

# Create a feature for the number of years since publication and drop rows with missing values
df_combined['Years_Since_Publication'] = 2024 - df_combined['Year']
df_combined.dropna(subset=['Years_Since_Publication', 'Citation_Class'], inplace=True)

# Drop rows with any missing values in 'Years_Since_Publication' or 'Citation_Class'
df_combined = df_combined.dropna(subset=['Years_Since_Publication', 'Citation_Class'])

# Select your features and target variable
X = df_combined[['Years_Since_Publication']]
y = df_combined['Citation_Class']



# STEP 2: Text Preprocessing
# ===========================

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

df_combined['Text'] = df_combined['Title']  # Assuming Abstract is not used or missing
df_combined['Processed_Text'] = df_combined['Text'].apply(preprocess_text)


# STEP 3: Feature Engineering
# ===========================

# Split text into unigrams, bigrams, trigrams
def process_text(text):
    new_text = preprocess_text(text)
    tokens = word_tokenize(new_text)
    unigrams = tokens
    bigrams = [' '.join(bg) for bg in list(ngrams(tokens, 2))]
    trigrams = [' '.join(tg) for tg in list(ngrams(tokens, 3))]
    return unigrams, bigrams, trigrams

df_combined[['Unigrams', 'Bigrams', 'Trigrams']] = df_combined['Text'].apply(process_text).apply(pd.Series)

df_combined['New_Unigrams'] = 0
df_combined['New_Bigrams'] = 0
df_combined['New_Trigrams'] = 0

# Initialize TF-IDF Vectorizer (though not used for fitting or transforming here)
tfidf_vectorizer = TfidfVectorizer()

# Function to calculate novelty scores using Euclidean distance on Count Vectorizer
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(df_combined['Processed_Text'])

def calculate_novelty_scores(matrix, k=5):
    euclidean_dist_matrix = pairwise_distances(matrix, metric='euclidean')
    sorted_distances = np.sort(euclidean_dist_matrix, axis=1)[:, 1:k+1]
    return np.mean(sorted_distances, axis=1)

df_combined['Processed_Text'] = df_combined['Title'].apply(preprocess_text)
df_combined['Novelty_Score'] = calculate_novelty_scores(count_matrix, k=5)


# STEP 4: Baseline Comparison
# ============================

# Load the baseline dataset and process text
baseline_file_path = "C:/Users/aadel/OneDrive/Documents/Mcgill Masters/3. Winter Term/Winter 1/Text Analytics/Group Project/Data/NLP_Papers_2015.csv"
df_baseline = pd.read_csv(baseline_file_path)
df_baseline['Text'] = df_baseline['Title']  # Assuming Abstract is not used or missing
df_baseline[['Unigrams', 'Bigrams', 'Trigrams']] = df_baseline['Text'].apply(process_text).apply(pd.Series)

# Extract baseline n-grams
baseline_unigrams = [unigram for row in df_baseline['Unigrams'] for unigram in row]
baseline_bigrams = [bigram for row in df_baseline['Bigrams'] for bigram in row]
baseline_trigrams = [trigram for row in df_baseline['Trigrams'] for trigram in row]

# Function to count new words
def count_new_words(ngrams, baseline_ngrams):
    return len(set(ngrams) - set(baseline_ngrams))

df_combined['New_Unigrams'] = df_combined['Unigrams'].apply(lambda x: count_new_words(x, baseline_unigrams))
df_combined['New_Bigrams'] = df_combined['Bigrams'].apply(lambda x: count_new_words(x, baseline_bigrams))
df_combined['New_Trigrams'] = df_combined['Trigrams'].apply(lambda x: count_new_words(x, baseline_trigrams))


# STEP 5: Model Building and Evaluation
# ======================================

# Model 1: Using only 'Years_Since_Publication'
X1 = df_combined[['Years_Since_Publication']]
y = df_combined['Citation_Class']
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42, stratify=y)
log_reg1 = LogisticRegression(random_state=42)
log_reg1.fit(X1_train, y_train)
y_pred1 = log_reg1.predict(X1_test)
print("Model 1 Confusion Matrix:\n", confusion_matrix(y_test, y_pred1))
print("Model 1 Classification Report:\n", classification_report(y_test, y_pred1))

# 5.2 Model 2: Using 'Years_Since_Publication' and Novelty Scores Variations
df_combined['Novelty_Scores_Log'] = np.log(df_combined['Novelty_Score'] + 1)
df_combined['Novelty_Scores_Squared'] = df_combined['Novelty_Score'] ** 2
features = ['Years_Since_Publication', 'Novelty_Score', 'Novelty_Scores_Log', 'Novelty_Scores_Squared']
X2 = df_combined[features]
scaler = StandardScaler()
X2_scaled = scaler.fit_transform(X2)
X2_train, X2_test, y_train, y_test = train_test_split(X2_scaled, y, test_size=0.2, random_state=42, stratify=y)
log_reg2 = LogisticRegression(random_state=42)
log_reg2.fit(X2_train, y_train)
y_pred2 = log_reg2.predict(X2_test)
conf_matrix2 = confusion_matrix(y_test, y_pred2)
class_report2 = classification_report(y_test, y_pred2)
print("Model 2 Confusion Matrix:\n", conf_matrix2)
print("Model 2 Classification Report:\n", class_report2)

# 5.3 Model 3: Including New Unigrams, Bigrams, Trigrams
features2 = ['Years_Since_Publication', 'Novelty_Score', 'Novelty_Scores_Log', 'Novelty_Scores_Squared', 'New_Unigrams', 'New_Bigrams', 'New_Trigrams']
X3 = df_combined[features2]
X3_scaled = scaler.fit_transform(X3)
X3_train, X3_test, y_train, y_test = train_test_split(X3_scaled, y, test_size=0.2, random_state=42, stratify=y)
log_reg3 = LogisticRegression(random_state=42)
log_reg3.fit(X3_train, y_train)
y_pred3 = log_reg3.predict(X3_test)
conf_matrix3 = confusion_matrix(y_test, y_pred3)
class_report3 = classification_report(y_test, y_pred3)
print("Model 3 Confusion Matrix:\n", conf_matrix3)
print("Model 3 Classification Report:\n", class_report3)


# STEP 6: Prediction Pipeline for New Papers Using Model 3
# ==========================================
warnings.filterwarnings('ignore', message='X does not have valid feature names')

def predict_new_paper(title, year):
    processed_title = preprocess_text(title)
    title_vector = count_vectorizer.transform([processed_title])
    novelty_score = calculate_novelty_scores(np.vstack((count_matrix.toarray(), title_vector.toarray())))[-1]
    novelty_score_log = np.log(novelty_score + 1)
    novelty_score_squared = novelty_score ** 2
    years_since_publication = 2024 - year
    unigrams, bigrams, trigrams = process_text(processed_title)
    new_unigrams = count_new_words(unigrams, baseline_unigrams)
    new_bigrams = count_new_words(bigrams, baseline_bigrams)
    new_trigrams = count_new_words(trigrams, baseline_trigrams)
    features = np.array([[years_since_publication, novelty_score, novelty_score_log, novelty_score_squared, new_unigrams, new_bigrams, new_trigrams]])
    features_scaled = scaler.transform(features)
    predicted_class = log_reg3.predict(features_scaled)
    
    # Organized output
    print("Paper Title: '{}'".format(title))
    print("Predicted Class: {}".format(predicted_class[0]))
    print("Novelty Score: {:.2f}".format(novelty_score))
    print("New Unigrams: {}, New Bigrams: {}, New Trigrams: {}".format(new_unigrams, new_bigrams, new_trigrams))
    print("-" * 60)  # Separator for readability

# Example usage
predict_new_paper("Natural Language Processing review", 2018)
predict_new_paper("jack jack jack jack jack jack jack jack jack jack jack", 2018)
predict_new_paper("Transplantation of cultured islets from two-layer preserved pancreases", 2018)
