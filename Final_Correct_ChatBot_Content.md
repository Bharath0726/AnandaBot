
# ChatBot: Master Crypto with Ananda

Welcome to the OPIM5770-Team5/ChatBot repository.

## Introduction

The Ananda AI chatbot is a sophisticated AI-powered tool developed to assist users with cryptocurrency and DeFi-related queries. Built with cutting-edge language models and designed with both performance and scalability in mind, this chatbot bridges the gap between complex financial concepts and end-user understanding.

Our chatbot leverages state-of-the-art AI technologies to provide:
- Real-time, accurate, and context-aware responses.
- Document-based assistance using retrieval-augmented generation (RAG).
- A user-friendly interface for seamless interaction.

This README explains the setup, functionality, and technical details of the chatbot project to ensure ease of understanding and usability for both technical and non-technical users.

---

## Project Folder Structure

```plaintext
Chatbot/
├── app.py                           # Main application file for the chatbot
├── environment.py                   # File to load environment variables
├── .env                             # Environment configuration file
├── requirements.txt                 # Dependencies required to run the project
├── docs/                            # Folder for RAG-based document storage
├── llm_metrics_evaluation.py        # Evaluation script for assessing model performance
├── T-Test Analysis for LLM Comparisons.py # Script for T-test analysis
├── Weighted average of all models.csv     # Weighted average results for all models
├── Evaluation Metrics results.csv         # Complete evaluation metrics for all models
└── T-Test Results.pdf                     # Final T-test results for model comparisons
```

---

## Technologies Used

### Programming Languages
- **Python**: Core programming language used for all development.

### Frameworks and Libraries
- **LangChain**: Orchestrates LLMs and integrates retrieval-augmented generation (RAG) for document-based querying.
- **Streamlit**: Provides a user-friendly web interface for the chatbot.
- **FAISS (Facebook AI Similarity Search)**: Implements fast, vector-based similarity search for RAG.
- **Transformers (Hugging Face)**: Used for pre-trained embeddings and fine-tuning.
- **pandas**: Manages data analysis and manipulation.
- **scikit-learn**: Computes cosine similarity for semantic comparisons.
- **NLTK**: Calculates BLEU scores for n-gram overlaps.
- **ROUGE Score**: Assesses recall-based similarity.
- **BERTScore**: Evaluates contextual similarity between generated and reference answers.
- **dotenv**: Securely manages environment variables.

### APIs
- **Groq API**: Powers the integration with Groq LLMs (e.g., Llama3, Mixtral, and Gemma models).
- **Hugging Face API**: Retrieves pre-trained embeddings for semantic similarity.
- **LangChain API**: Enables LLM orchestration and tracking.

### Tools and Platforms
- **Microsoft Excel/Google Sheets**: For tabular data organization and analysis.
- **Streamlit Hosting**: Provides a web interface for deploying the chatbot.
- **FAISS**: Optimizes document retrieval for RAG.

---

## Setup Instructions

### Step 1: Prerequisites
Before starting, ensure the following:
- Python 3.8 or higher installed on your system.
- A virtual environment tool like `venv` or `conda`.
- API keys for Groq, LangChain, and Hugging Face.

### Step 2: Installation

#### Clone the Repository:
```bash
git clone https://github.com/your-repo/ananda-chatbot.git
cd ananda-chatbot
```

#### Install Dependencies:
Use the following command to install all required Python libraries:
```bash
pip install -r requirements.txt
```

#### Configure API Keys:
Create a `.env` file in the root directory and add your API keys:
```plaintext
LANGCHAIN_API_KEY=your_langchain_api_key
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
LANGCHAIN_PROJECT=your_project_name
```

#### Add Documents (Optional):
If using the RAG configuration, add your PDF files to the `docs/` folder. These files will be processed and used to answer document-specific questions.

#### Run the Application:
Launch the chatbot using Streamlit:
```bash
streamlit run app.py
```
Open the provided URL in your browser to interact with the chatbot.

---

## How It Works

### Chatbot Workflow
1. **User Interaction**: The user inputs a query through a simple and interactive chat interface.
2. **Base or RAG Configuration**:
   - If documents are uploaded to the `docs/` folder, the chatbot retrieves relevant content using FAISS and Hugging Face embeddings.
   - If no documents are available, the chatbot relies solely on the Llama 3 model to generate responses.
3. **Response Generation**: The chatbot processes the query and provides an answer based on context.

---


## Code Explanation and File Description

### 1. `app.py`
**Purpose**: This is the main script to launch the chatbot. It integrates user interaction, LLM (Large Language Model) responses, and optional document retrieval for RAG (Retrieval-Augmented Generation)-based queries.

**Detailed Explanation**:
- **Libraries Imported**:
  ```python
  import streamlit as st
  import os
  from dotenv import load_dotenv
  from langchain.chains import LLMChain
  from langchain.prompts import ChatPromptTemplate
  from langchain.llms import OpenAI
  from langchain.vectorstores import FAISS
  from langchain.document_loaders import PyPDFDirectoryLoader
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  ```
- **Functionality**:
  - Streamlit creates the user-friendly interface.
  - LangChain orchestrates LLM responses and integrates RAG.
  - FAISS enables document retrieval for enhanced context.

---

### 2. `environment.py`
**Purpose**: Loads environment variables required for the chatbot's configuration and API access.

**Code**:
```python
import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    return {
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
        "HF_TOKEN": os.getenv("HF_TOKEN"),
    }
```

---

### 3. `llm_metrics_evaluation.py`
**Purpose**: Evaluates the performance of three LLMs (Llama, Mixtral, and Gemma) using various metrics.

**Detailed Explanation**:
- **Libraries Imported**:
  ```python
  import pandas as pd
  from sklearn.metrics.pairwise import cosine_similarity
  from nltk.translate.bleu_score import sentence_bleu
  from rouge_score import rouge_scorer
  ```

- **Workflow**:
  - **Data Loading**:
    - Loads CSV files containing:
      - Gold-standard answers.
      - Answers generated by Llama, Mixtral, and Gemma.
  - **Metrics**:
    - BLEU, ROUGE-L, Cosine Similarity, BERTScore, Accuracy.
  - **Output**:
    - Results are saved in `comparison_results_weighted_RAG.csv`.

---

### 4. `T-Test Analysis for LLM Comparisons.py`
**Purpose**: Performs statistical tests (paired T-tests) to determine significant differences between the models.

**Code**:
```python
from scipy.stats import ttest_rel

# Example function for paired T-test
def perform_ttest(model_a_scores, model_b_scores):
    t_stat, p_value = ttest_rel(model_a_scores, model_b_scores)
    return t_stat, p_value
```

---

### 5. `Weighted average of all models.csv`
**Purpose**: Contains the weighted average scores of all models for 96 questions.

**Detailed Explanation**:
- **Columns Include**:
  - Weighted averages for Base Llama, Mixtral, Gemma, and their RAG versions.

---

### 6. `Evaluation Metrics results.csv`
**Purpose**: Stores detailed metrics (BLEU, ROUGE-L, Cosine Similarity, bert score, accuracy) for each model and question.

---

### 7. `T-Test Results.pdf`
**Purpose**: Summarizes the results of all model comparisons with tables for:
- T-statistic values.
- P-values.
- Statistical significance.


## Evaluation Workflow

1. **Chatbot Setup:**
   - Launch `app.py` to initialize the chatbot.
   - Interact with the chatbot and analyze responses.

2. **Model Evaluation:**
   - Use `llm_metrics_evaluation.py` to compute performance metrics.
   - Save results to `comparison_results_weighted_RAG.csv`.

3. **Statistical Analysis:**
   - Run `T-Test Analysis for LLM Comparisons.py` for detailed comparisons.
   - Review `T-Test Results.pdf` for insights.

---

## Key Features of the Chatbot

- **LLM-Driven Responses:** Powered by Groq's Llama3, optimized for cryptocurrency-related queries.
- **Document Retrieval:** Incorporates RAG-based techniques for improved context and reliability.
- **User-Friendly Interface:** Interactive UI built with Streamlit for smooth communication.
- **Evaluation Metrics:** Robust system for assessing model performance using BLEU, ROUGE, and BERTScore.

---

## Contact

For questions, suggestions, or collaboration opportunities:  
- **Email**: team5@opim5770.com  
- **GitHub Repository**: [Ananda Chatbot](https://github.com/your-repo/ananda-chatbot)

---
