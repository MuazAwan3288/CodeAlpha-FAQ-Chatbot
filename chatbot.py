# FAQ chatbot Uses NLP preprocessing, TF-IDF vectorization, and cosine similarly to find the bestt FAQ answer.

import json 
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.text_processing import preprocess_text

class FAQChatbot:
    """FAQ chatbot that matches user questions with the most similar FAQ."""
    def __init__(self, faq_file = "data/faqs.json", threshold = 0.20):
# Initialize the Chatbot Args: faq_file (str): Path to the FAQ Json file.
# Threshold (float): minimum similarly required to return an FAQ answer           
        self.faq_file = Path(faq_file)
        self.threshold = threshold

# Load FAQ data.
        self.faqs = self.load_faqs()

        if not self.faqs:
            raise ValueError("No FAQs were found in the datasheet.")

# Store question and answers sseparately.
        self.questions = [faq["question"] for faq in self.faqs]    
        self.answers = [faq["answer"] for faq in self.faqs]

# Create the TF-IDF vectorizer.
        self.vectorizer = TfidfVectorizer(preprocessor = preprocess_text, tokenizer = str.split, token_pattern = None, lowerxase = False)

# Convert all FAQsquestions into TF-IDf vectors.
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

def load_faqs(self):
    """Load FAQs from the JSON file.
    Returns:
        list: List of FAQs dictionaries.
    """
    if not self.faq_file.exists():
        raise FileNotFoundError(f"FAQ file not found: {self.faq_file}")
    with self.faq_file.open("r", encoding = "utf-8") as file:
        data = json.load(file)

# Validata the dataset.
    if not isinstance(data, list):
        raise ValueError("FAQ JSON must contain a list of question and answers.")

    for index, faq in enumerate(data):
        if "question" not in faq or "answer" not in faq:
            raise ValueError(f"FAQ item {index + 1} must contain 'question' and 'answer'")

    return data

def get_response(self, user_question):

    """
    Find the most similar FAQ and return its answer.
    Args:
        user_question (str): Question entered by the user.
    Returns:
        tuple: (answer, similarity_score, matched_question)
    """
# Validate input
    if not user_question or not user_question.strip():
        return ("Pleaase enter a question so I can help you.", 0.0, "")

# Convert user question into a TK-IDF vector.
    user_vector = self.vectorizer.transform([user_question])

# Calculate cosine similarity against all FAQ questions.
    similarity_score = cosine_similarity(user_vector, self.question_vectors)[0]

# Find the index of the highest score.
    best_index = similarity_score.argmax()

# Get the highest similariy score.
    best_score = float(similarity_score[best_index])

# Get the corresponding FAQ.
    best_question = self.questions[best_index]
    best_answer = self.answers[best_index]

# If similarity is to low, don't give an unrelated answer.
    if best_score < self.threshhold:
        return ("Sorry I couldn't find a relevant answer to your question."
        "Please try asking about AI, Machine Learning, NLP, Python, Chatbot"
        "TF-IDF, cosine similarity, or relatedd topisc.", best_score, "")
    return (best_answer, best_score, best_question)

def respond(self, user_question):
# Create a user-friendly chatbot response.
    answer, score, matched_question = self.get_response(user_question)    

# Show the matching information for transparency.
    if matched_question:
        return (f"{answer}\n\n" f"**SWimilarity score:** {score:.2f}\n"
                f"**Matched FAQ:** {matched_question}")
    return answer

if __name__ == "__main__":
# Test the chatbot from the terminal
    chatbot = FAQChatbot()    
    print("FAQ Chatbot is ready!")
    print("Type 'exit' to stop.\n")
    while True:
        question = input("you: ")
        if question.lower().strip() == "exit":
            print("Bot: Gooodbye!")
            break

        print("Bot: ", chatbot.respond(question))
        print()
        