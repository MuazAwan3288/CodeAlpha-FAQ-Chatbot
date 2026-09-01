# 🤖 CodeAlpha FAQ Chatbot

An AI-powered FAQ chatbot developed as part of the CodeAlpha Artificial Intelligence Internship.

The chatbot uses Natural Language Processing (NLP), TF-IDF vectorization, and cosine similarity to match user questions with the most relevant FAQ and return an appropriate answer.

## 📌 Project Overview

This project demonstrates how NLP techniques can be used to build a simple FAQ chatbot without requiring a large language model.

The chatbot contains a collection of frequently asked questions related to:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Natural Language Processing
- Python
- Chatbots
- TF-IDF
- Cosine Similarity
- NLTK
- Scikit-learn
- Gradio

## ✨ Features

- FAQ-based chatbot
- NLP text preprocessing
- Lowercase conversion
- Tokenization
- Stop-word removal
- Stemming using NLTK
- TF-IDF vectorization
- Cosine similarity matching
- Similarity threshold
- Best FAQ detection
- Interactive Gradio interface
- 30 FAQ questions and answers
- Unknown-question fallback response

## 🧠 How It Works

The chatbot follows this process:

User Question
↓
Text Preprocessing
↓
TF-IDF Vectorization
↓
Cosine Similarity
↓
Find Most Similar FAQ
↓
Check Similarity Threshold
↓
Return Best Answer

## 🛠️ Technologies Used

- Python
- NLTK
- Scikit-learn
- Gradio
- JSON

## 📁 Project Structure

```text
CodeAlpha-FAQ-Chatbot/
│
├── .gitignore
├── README.md
├── requirements.txt
├── app.py
├── chatbot.py
│
├── data/
│   └── faqs.json
│
├── utils/
│   ├── __init__.py
│   └── text_processing.py
│
└── assets/
    └── icons/
        └── .gitkeep