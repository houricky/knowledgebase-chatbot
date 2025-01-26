from langchain.document_loaders import PyPDFLoader
from pypdf import PdfReader
import tempfile
import os
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_chat import message
import docx2txt, sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def document_loader(doc):
    """
    :param doc: document path
    :return: document object
    """
    loader=PyPDFLoader(doc)
    document_2=loader.load()
    return document_2

def read_docx(docFileObj):
    '''
    :param: take docFileObj as input document object
    :return: the read text as output
    '''
    all_text = docx2txt.process(docFileObj)
    return all_text

def pdf_reader(pdf_doc):
    """
    :param pdf_doc:
    :return: pdf text
    """
    pdf_reader=PdfReader(pdf_doc)
    text = ""
    pages=pdf_reader.pages
    print(len(pages))
    for page in pages:
        text += page.extract_text()
    return text,len(pages)

def pagetext_pagenum(pdf_doc):
    """
    :param pdf_doc:
    :return: page numbers list and page texts list
    """
    pdf_reader=PdfReader(pdf_doc)
    pages=pdf_reader.pages
    page_texts = []
    page_nos = []
    i=1
    for page in pages:
        page_txt = page.extract_text()
        page_no = i
        page_texts.append(page_txt)
        page_nos.append(page_no)
        i+=1
    return page_texts, page_nos

def find_pagenum(corpus, input_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    input_tfidf = vectorizer.transform([input_text])
    cosine_similarities = cosine_similarity(input_tfidf, tfidf_matrix)
    most_similar_corpus_index = cosine_similarities.argmax()
    return most_similar_corpus_index

def creating_temp_file(doc_name):
    """
    :param doc_name: Name of the document
    :return: path of the document in temporary directory
    """
    temp_file_path = os.getcwd()
    temp_dir = tempfile.TemporaryDirectory()
    print(temp_file_path)
    temp_file_path = os.path.join(os.getcwd(), temp_dir.name.split("\\")[len(temp_dir.name.split("\\"))-1], doc_name.name)
    print(temp_file_path)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(doc_name.read())
    return temp_file_path


def menu_bar_chatbot(idx):
    selected=option_menu(None, ["Talk to docs", "Talk to data"],
                icons=["filetype-pdf", "database-fill-gear"],
                menu_icon="app-indicator", orientation="horizontal", default_index=idx,
                styles={
                    "container": {"padding": "0!important", "background-color": "#E0FFFF"},
                    "icon": {"color": "orange", "font-size": "14px"},
                    "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px",
                                 "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "#4682B4"},
                }
                )
    return selected


def display_chat(chat_history):
    for i, chat in enumerate(reversed(chat_history)):
        if "user" in chat:
            message(chat["user"], is_user=True, key=str(i)) 
        else:
            message(chat["bot"], key="bot_"+str(i))
