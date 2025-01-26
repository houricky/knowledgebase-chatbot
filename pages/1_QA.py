from langchain.callbacks import get_openai_callback
from PIL import Image
import pandas as pd
import streamlit as st
from src.llm_models import AzureOpenAI
from src.Q_and_A_model import q_and_a_model, chat_with_csv
from utils.config_manager import get_config_by_key
from utils.util import pdf_reader, pagetext_pagenum
from src.database import PostgreSQLCRUD
import datetime
import numpy as np
from utils.audit_manager import Send_to_Audit_ask
from utils.embedding_cost import Text_to_Embedding_Conversion_cost
import langchain
import logging
langchain.verbose = False

try:
    llm_obj = AzureOpenAI()
    llm = llm_obj.azure_open_ai_gpt35()
    
    embeddings = llm_obj.azure_open_ai_embedding_gpt35()
    
    conn = PostgreSQLCRUD()
    
    encoding_model = get_config_by_key('embeddings_cost','encoding_model')

    st.set_page_config(page_title="Chatbot", page_icon="📝", layout='wide')
    n=0

    st.sidebar.title(":blue[Chatbot]")
    st.sidebar.header(":blue[Ask to doc]")
    st.sidebar.markdown(f"for quick demo download the the file for Ask the document use-case from following link")

    st.title(':blue[Knowledgebase chatbot]')

    selected = menu_bar_LS_Ask(0)
    logging.info(f'Menu bar selection made: {selected}')

    if selected == 'Talk to docs':
        create_datetime = datetime.datetime.now()
        st.title("📝 Talk to Pdf ")
        try:
            uploaded_file = st.file_uploader("**:blue[Upload a pdf]**", type="pdf")
            if uploaded_file is not None:
                logging.info(f"PDF file uploaded: {uploaded_file.name}")
                pdf_text,page_count = pdf_reader(uploaded_file)
                logging.info(f"PDF processed. Page count: {page_count}")
                page_texts, page_nos = pagetext_pagenum(uploaded_file)
                bytes_data = uploaded_file.getvalue()
                file_size  = len(bytes_data)

                if page_count <= int(c_page_count):
                    if pdf_text:
                        logging.info("Displaying document text and awaiting user queries.")
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            with st.expander("**:blue[View document text]**"):
                                st.text(pdf_text)
                        with col2:
                            qa_chain = q_and_a_model(llm, embeddings, pdf_text)
                            query_text = st.text_area("**:blue[Enter your question:]**",
                                                    placeholder='Please provide a short summary.',
                                                    disabled=not uploaded_file)
                            col1, col2 = st.columns([0.8, 0.2])
                            with col2:
                                submit = st.button(":orange[Submit]")
                            if submit:
                                st.session_state.feedback_key=st.session_state.feedback_key+1
                                logging.info("User submitted a query.")
                                with st.spinner('Processing...'):
                                    with get_openai_callback() as cb:
                                        num_tokens, embedding_cost = Text_to_Embedding_Conversion_cost(pdf_text,encoding_model, cost_model)
                                        response = qa_chain({"query":query_text})
                                        logging.info("Response generated for the user query.")
                                        st.write(F"Response: {response['result']}")
                                                                                
                                        with st.expander("**:blue[View citations]**"):
                                            for idx, elt in enumerate(response['source_documents']):
                                                citation = elt.page_content
                                                citation_pg_idx = find_pagenum(page_texts, citation)
                                                citation_page_no = page_nos[citation_pg_idx]
                                                st.subheader("Citation", divider='rainbow')
                                                st.write(f"Page No: {citation_page_no}")
                                                st.write(f"        Content: {elt.page_content}")
                                        total_cost = cb.total_cost
                                        with st.expander("**:blue[View cost]**"):
                                            st.write("Number of Tokens for converting to Embeddings:", num_tokens)
                                            st.write("Embedding Conversion cost:", embedding_cost)
                                            st.write(cb)
                                            st.write("actual cost from callback:", total_cost)
                                            total_cost = total_cost + embedding_cost
                                            st.write("total cost:", total_cost)

                                        prompt_token = cb.prompt_tokens
                                        completion_token = cb.completion_tokens
                                        #total_cost = cb.total_cost
                                        file_name = uploaded_file.name
                                        update_datetime = datetime.datetime.now()
                                        logging.info("Displaying cost and citation information.")
                                        args = ("llm_pg_ls_ask", user_id,selected, file_name,file_size,prompt_token,completion_token,total_cost,create_datetime,update_datetime,query_text,conn)
                                        Send_to_Audit_ask(*args)
                                        logging.info("Audit information sent.")
                    else:
                        st.error("Pdf is not readable")
                        logging.error("Pdf is not readable")
                else:
                    st.error(f"Pdf is having more than {c_page_count} pages")
                    logging.error(f"Pdf is having more than {c_page_count} pages")
        except Exception as e:
            st.error(f"Unable to process this pdf {e}")
            logging.error(f"Unable to process this pdf {e}")

        rows = conn.read_records_with_filter("llm_pg_ls_ask",["id","user_id",'ask_type','args','create_datetime'],"user_id",user_id)
        rows = rows[rows.ask_type == "Talk to docs"]
        if rows.shape[0] > 0:
            logging.info("Records found for 'Talk to docs'. Processing to find the latest record.")
            latest_record = rows[rows.id == rows.id.max()]
            latest_record_id = latest_record['id'].to_list()[0]
            latest_record_feature = latest_record['ask_type'].to_list()[0]
            latest_record_args = latest_record['args'].to_list()[0]
            latest_record_time = latest_record['create_datetime'].to_list()[0]
            latest_record_time = latest_record_time.strftime('%Y-%m-%d %H:%M')
            create_datetime = create_datetime.strftime('%Y-%m-%d %H:%M')
            if latest_record_feature =="Talk to docs" and latest_record_args is None and create_datetime == latest_record_time:
                logging.info(f"Conditions met for updating record ID: {latest_record_id} with new arguments.")
                feedback_json = Send_to_Audit_thumbs_fb(st.session_state.feedback_key)
                if feedback_json:
                    conn.update_record("llm_pg_ls_ask", "args", feedback_json, "id", latest_record_id)
                    logging.info(f"Record ID: {latest_record_id} updated successfully.")
        else:
            logging.info("No 'Talk to docs' records found for the given user_id.")
    
        
except Exception as e:
    logging.error("An unexpected error occurred in the Ask functionality.", exc_info=True)
    st.error(f"An unexpected error occurred: {e}")
finally:
    conn.close_connection()
    logging.info("Database connection closed.")