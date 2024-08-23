import streamlit as st
import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from htmlTemplate import css, bot_template

files = ['/Users/jeremyquek/Downloads/LTA_AR2223.pdf',
                     '/Users/jeremyquek/Downloads/LTA_SR2223.pdf',
                     '/Users/jeremyquek/Downloads/LTA_ERP_2.pdf',
                     '/Users/jeremyquek/Downloads/SAF_Handbook.pdf'
                     ]

def load_documents(pdf_file_path):
    print("Loading documents...")
    loaded_file_list = []
    for file in pdf_file_path:
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        loaded_file_list.append(pages)
        print(len(loaded_file_list))
    return loaded_file_list

def create_vectorstore(dataset):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    for index, data in enumerate(dataset):
        if index == 0:
            print("Embedding document...")
            vector_store = FAISS.from_documents(data, embeddings)
        else:
            print("Embedding document...")
            faiss_index = FAISS.from_documents(data, embeddings)
            vector_store.merge_from(faiss_index)
    return vector_store

def create_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.5, model_name="gpt-4o")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Introduce yourself as an assistant chatbot.\
      Answer the user's question based SOLELY on the given context: {context}\
      You are to answer only from the context.\
      If you are unsure or unable to answer from context, say you don't know and do not make things up.\
      Always provide your source when answering from the source in the format (Source: <name of source document>) at the END of the conversation.\
      If no sources are used to answer, do not cite them"
         ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )
    conversation_chain = create_retrieval_chain(
        vectorstore.as_retriever(),
        chain,
    )
    return conversation_chain

def gen_answer(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    chat_history.append(HumanMessage(content=query))
    return response["answer"]


def streamlitUI(x):
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    st.title("RAG CHAT BOT")
    st.header("Chat with multiple documents")
    st.write(
        f"I am an assistant chatbot here to engage in friendly conversation, I am grounded with the following {x}documents: ")
    st.write("- LTA Annual Report 2022/2023")
    st.write("- LTA Sustainability Report 2022/2023")
    st.write("- LTA ERP 2.0")
    st.write("- SAF Professional Reading List")
    user_question = st.text_input("Feel free to ask me any question on these topics!")

    return user_question

def handle_query():
    st.write(css, unsafe_allow_html=True)
    with st.spinner("Generating..."):
        st.session_state.conversation = gen_answer(chain, query, chat_history)
    response = st.session_state.conversation
    st.write(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)

if __name__ == "__main__":

    chat_history = []

    dataset = load_documents(files)

    vectorstore = create_vectorstore(dataset)

    chain = create_chain(vectorstore)

    x =5
    query = streamlitUI(x)

    if query:
        handle_query()












# streamlit run localapp.py


