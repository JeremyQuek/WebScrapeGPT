# RAG ChatBot with Streamlit UI

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Streamlit for the user interface. The chatbot is capable of answering questions based on multiple PDF documents.

## Features

- Load and process multiple PDF documents
- Create embeddings and vector store using OpenAI's embedding model
- Implement a conversational chain using LangChain and OpenAI's GPT-4 model
- Streamlit-based user interface for easy interaction

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone this repository
2. Install the required packages:
3. Set your OpenAI API key as an environment variable or directly in the code

## Usage

1. Update the `files` list with the paths to your PDF documents
2. Run the Streamlit app:
3. Interact with the chatbot through the web interface

## Configuration

- Adjust the `ChatOpenAI` parameters in the `create_chain` function to modify the LLM behavior
- Customize the system prompt in the `create_chain` function to change the chatbot's personality and instructions

## Files

- `localapp.py`: Main application file
- `htmlTemplate.py`: Contains CSS and HTML templates for the UI (not provided in the snippet)

## Note

- The current implementation loads and processes documents on each run. For production use, consider implementing caching or pre-processing of documents.
- GPU acceleration and document insertion/embedding improvements are noted as potential enhancements.

## TODO

- Implement conversation history persistence
- Improve handling of questions about unrelated topics
- Optimize GPU usage for faster processing
- Enhance document insertion and embedding efficiency
