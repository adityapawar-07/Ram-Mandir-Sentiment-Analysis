from flask import Flask, render_template, request
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("models/Best_Model_Random_Forest.pkl")  # Replace with your model file
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")  # Load the TF-IDF vectorizer

# Initialize NLTK tools
nltk.download('stopwords')
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Preprocess the input text (same as in training)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Clean the text
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase and strip whitespace
    # Tokenize and stem
    tokens = text.split()
    processed_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(processed_tokens)

# Route to handle the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle sentiment prediction
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        user_comment = request.form["comment"]  # Get user input
        preprocessed_comment = preprocess_text(user_comment)  # Preprocess the comment
        vectorized_comment = tfidf_vectorizer.transform([preprocessed_comment])  # Vectorize the input
        
        # Predict sentiment using the trained model
        sentiment = model.predict(vectorized_comment)[0]

        # Render the result in the HTML template
        return render_template("index.html", comment=user_comment, sentiment=sentiment)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
