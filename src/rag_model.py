from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.vectorstores.pgvector import PGVector
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from pandasai import PandasAI
from matplotlib import pyplot as plt
import os
import streamlit as st
from langchain.callbacks import get_openai_callback
from utils.config_manager import  get_postgres_config_secret
import langchain
langchain.verbose = False
# initialize config variables
db_config = get_postgres_config_secret()

def rag_model(llm,embeddings,uploaded_file):
    connection_string = db_config['connection_string']
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100,length_function=len)
    splitted_text = text_splitter.split_text(uploaded_file)
    db = Chroma.from_texts(splitted_text, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True)
    return qa

