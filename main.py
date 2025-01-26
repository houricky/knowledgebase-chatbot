import pandas as pd
from streamlit_option_menu import option_menu

import streamlit as st
from src.llm_models import AzureOpenAI
from src.database import Database
from src.chat_model import chat_bot
from src.Q_and_A_model import q_and_a_model
from utils.config_manager import get_config_by_key
from utils.util import pdf_reader

st.set_page_config(layout='wide')

llm_obj = AzureOpenAI()
llm = llm_obj.azure_open_ai_gpt35()
embeddings = llm_obj.azure_open_ai_embedding_gpt35()

def option_menu_selection(idx):
    st.session_state.idx=idx

def menu_bar(idx):
    selected=option_menu(None, ["Home", "chatbot", "Ask your knowledgebase"],
                icons=['house', 'chat-text', 'chat-text-fill', "filetype-pdf", "database-fill-gear"],
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

if selected == 'Home':

    st.header('This is the home page')

elif selected == 'chatbot':
    CHAT_INTERACTION_TO_KEEP = get_config_by_key('chat_model', 'CHAT_INTERACTION_TO_KEEP')
    st.title("your knowledgebase chatbot")
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        if "interaction_count" not in st.session_state:
            st.markdown(f":green[Interaction Count 0]")
        else:
            st.markdown(f":green[Interaction Count {st.session_state.interaction_count}]")

    col1, col2 = st.columns([0.8,0.2])
    with col2:
        refresh = st.button(":orange[Refresh]")
        if refresh:
            st.session_state.clear()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "interaction_count" not in st.session_state:
        st.session_state["interaction_count"] = 0

    for msg in st.session_state.messages:
        message=st.chat_message(msg["role"])
        message.write(msg["content"])


    if prompt := st.chat_input("Hey whats up?"):

        st.chat_message("user").write(prompt)
        conversation_chain=chat_bot(llm)
        output=conversation_chain.predict(history =' '.join([f'role: {m["role"]}, content: {m["content"]}' for m in st.session_state.messages]),input=prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": output})
        st.session_state.interaction_count = st.session_state.interaction_count+1
        if CHAT_INTERACTION_TO_KEEP == str(st.session_state.interaction_count):
            st.session_state.clear()
        st.chat_message("assistant").write(output)


elif selected == 'Ask your knowledgebase':
    st.title("📝 ASk your knowledgebase ")
    uploaded_file=st.file_uploader("**:blue[Upload a pdf]**",type="pdf")
    if uploaded_file is not None:
        pdf_text = pdf_reader(uploaded_file)
        if pdf_text:
            col1, col2 = st.columns([1,1])
            with col1:
                with st.expander("**:blue[View document text]**"):
                    st.text(pdf_text)
            with col2:
                qa_chain=q_and_a_model(llm,embeddings,pdf_text)
                query_text = st.text_area("**:blue[Enter your question:]**", placeholder='Please provide a short summary.',
                                           disabled=not uploaded_file)
                col1,col2 = st.columns([0.8,0.2])
                with col2:
                    submit = st.button(":orange[Submit]")
                if submit:
                    with st.spinner('Processing...'):
                        response = qa_chain.run(query_text)
                        st.write(response)
        else:
            st.error("Pdf is not readable")










