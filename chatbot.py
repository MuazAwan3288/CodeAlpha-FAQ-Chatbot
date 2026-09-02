"""
FAQ Chatbot

Uses:
- NLP text preprocessing
- TF-IDF vectorization
- Cosine similarity

to find the most relevant FAQ answer.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.text_processing import preprocess_text


class FAQChatbot:
    """FAQ chatbot using TF-IDF and cosine similarity."""

    def __init__(self, faq_file="data/faqs.json", threshold=0.25):
        """Initialize the FAQ chatbot."""

        self.faq_file = Path(faq_file)
        self.threshold = threshold

        # Load FAQ data.
        self.faqs = self.load_faqs()

        # Store questions and answers separately.
        self.questions = [
            faq["question"] for faq in self.faqs
        ]

        self.answers = [
            faq["answer"] for faq in self.faqs
        ]

        # Create TF-IDF vectorizer.
        self.vectorizer = TfidfVectorizer(preprocessor = preprocess_text, tokenizer = str.split, token_pattern = None, lowercase = False)

        # Convert FAQ questions into TF-IDF vectors.
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def load_faqs(self):
        """Load and validate FAQs from the JSON file."""

        if not self.faq_file.exists():
            raise FileNotFoundError(f"FAQ file not found: {self.faq_file}")

        with self.faq_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("FAQ JSON must contain a list of FAQ objects.")

        for index, faq in enumerate(data):
            if not isinstance(faq, dict):
                raise ValueError(f"FAQ item {index + 1} must be an object.")

            if "question" not in faq or "answer" not in faq:
                raise ValueError(f"FAQ item {index + 1} must contain 'question' and 'answer'.")
            
        return data

    def get_response(self, user_question):
        """Find the most similar FAQ."""

        if not user_question or not user_question.strip():
            return ("Please enter a question so I can help you.", 0.0, "")

        # Convert user's question into a TF-IDF vector.
        user_vector = self.vectorizer.transform([user_question])

        # Calculate cosine similarity.
        similarity_scores = cosine_similarity(user_vector, self.question_vectors)[0]

        # Find the highest similarity score.
        best_index = similarity_scores.argmax()
        best_score = float(similarity_scores[best_index])

        best_question = self.questions[best_index]
        best_answer = self.answers[best_index]

        # Reject unrelated questions.
        if best_score < self.threshold:
            return (
                "Sorry, I couldn't find a relevant answer. "
                "Please try asking about AI, Machine Learning, "
                "NLP, Python, chatbots, TF-IDF, or cosine similarity.",
                best_score,
                ""
            )

        return (best_answer, best_score, best_question)

    def respond(self, user_question):
        """Return a formatted chatbot response."""

        answer, score, matched_question = self.get_response(user_question)

        if matched_question:
            return (
                f"{answer}\n\n"
                f"**Similarity score:** {score:.2f}\n"
                f"**Matched FAQ:** {matched_question}"
            )

        return answer


# Terminal testing
if __name__ == "__main__":

    chatbot = FAQChatbot()

    print("FAQ Chatbot is ready!")
    print("Type 'exit' to stop.\n")

    while True:

        question = input("You: ")

        if question.lower().strip() == "exit":
            print("Bot: Goodbye!")
            break

        print("Bot:", chatbot.respond(question))
        print()