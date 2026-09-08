import pandas as pd
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')

# Load FAQ dataset
data = pd.read_csv("faq.csv")

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Clean FAQ questions
data["clean_question"] = data["question"].apply(clean_text)

# Convert FAQ questions into TF-IDF vectors
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(data["clean_question"])

print("===================================")
print("       COLLEGE FAQ CHATBOT")
print("===================================")
print("Type 'exit' to stop the chatbot.")
print()

while True:

    user_question = input("You: ")

    # Exit chatbot
    if user_question.lower() == "exit":
        print("Bot: Thank you! Have a nice day.")
        break

    # Clean user question
    cleaned_question = clean_text(user_question)

    # Convert user question into vector
    user_vector = vectorizer.transform([cleaned_question])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    # Find best matching FAQ
    best_match_index = similarity_scores.argmax()

    # Get similarity score
    best_score = similarity_scores[0][best_match_index]

    # Set threshold
    if best_score >= 0.25:

        answer = data.iloc[best_match_index]["answer"]

        print("Bot:", answer)

    else:

        print("Bot: Sorry, I don't have an answer for that question.")

    print()