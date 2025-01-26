import streamlit as st
from utils.config_manager import get_config_by_key
from src.database import PostgreSQLCRUD
import logging
import sys
# Initialize logging configuration
trace_level = logging.INFO
# Configure the basic configuration for logging
logging.basicConfig(stream=sys.stderr, level=trace_level,
                    format='%(asctime)s %(levelname)s %(message)s')


st.set_page_config(
    page_title="Hello",
    page_icon="⚙️",
)
def ui_spacer(n=2, line=False, next_n=0):
    for _ in range(n):
        st.write('')
    if line:
        st.tabs([' '])
    for _ in range(next_n):
        st.write('')
        
def ui_info():
    
    try:
        app_name = get_config_by_key("application","APP_NAME")
        app_version = get_config_by_key("application","APP_VERSION")
        st.write("Welcome ", ss["token"]["account"]["name"])
        #st.write(ss["token"]["account"]["username"].split("@")[0])
        ui_spacer(1)
        st.write("Logged in successfully! Now let's Chat with the AI 🤖")
        ui_spacer(1)
        st.markdown(f"""
        # {app_name}
    	version {app_version}
        """)
    except Exception as e:
        st.write("An error occurred.")

ss = st.session_state

try:
    
    conn = PostgreSQLCRUD()
        
    # layout
    with st.sidebar:
        ui_info()
        ui_spacer(2)
    
        st.title(":blue[Welcome to My Knowledgebase Chatbot 👋]")
    
except Exception as e:
    st.write("An error occurred during authentication.")