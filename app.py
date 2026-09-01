# Gradio interface for the CodeAlpha FAQ Chatbot

import gradio as gr
from chatbot import FAQChatbot

# Create an Chatbot object once when thew application starts.
# This means the FAQ data and TF-IDF vectors are created only once.

chatbot = FAQChatbot()

def answer_question(question):

# Receive a question from the gradio interface and return the Chatbot response.
    return chatbot.respond(question)

# Create the Gradio application.
demo = gr.Interface(fn = answer_question, inputs = gr.Textbox(label = "Your Question",
        placeholder = "Ask a question about AI, Python, NLP, Machine Learning....", lines = 2),
        outputs = gr.Markdown(label = "Chatbot Response"),
        title = "🤖 CodeAlpha FAQ Chatbot",
        description = ("An NLP-powered FAQ chatbot using TF-IDF and cosine similarity to find the most relevent answer."),
        examples = [
            ["What is Artifical Intelligence?"],
            ["What is Machine Learning?"],
            ["What is NLP?"],
            ["What is Python popular for AI?"],
            ["What is cosine similarity?"],
            ["How does this chatbot find an answer?"]
        ],
        clear_btn = "Clear",
        submit_btn = "Ask"

        )

if __name__ == "__main__":
    print("Loading CodeAlpha FAQ Chatbot...")
    print(f"Loaded{len(chatbot.faqs)} FAQs.")
    print("Starting Gradio application...")

    demo.launch()