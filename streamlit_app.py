from API import api
import streamlit as st 
import pandas as pd
import json
import seaborn as sns
from Summary import summary_gen
import matplotlib.pyplot as plt
import io
import openai
from dataclasses import dataclass
from typing import Literal
import streamlit as st
import sys
import io

from langchain import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import streamlit.components.v1 as components

st.set_page_config(
    layout="wide", 
    page_title="EDA AUTOMATION", 
    page_icon="📂"
)
# Function to fetch raw CSS from GitHub
def github_css(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return f"<style>{response.text}</style>"
        else:
            return f"<style>/* Failed to load CSS: {response.status_code} */</style>"
    except Exception as e:
        return f"<style>/* Error loading CSS: {e} */</style>"

# URL to the raw CSS file on GitHub
css_url = "https://github.com/jaisai2512/Chat_data/blob/main/style.css"

# Inject the CSS into the Streamlit app
st.markdown(github_css(css_url), unsafe_allow_html=True)


# Set page layout

# Create a sidebar for file upload
st.sidebar.header("📁 Upload Your File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

# Display content on the main page
st.title("🤖 EDA Automation")

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    # Display file details
    st.write("### File Details:")
    file_details = {
        "Filename": uploaded_file.name,
        "Filetype": uploaded_file.type,
        "Filesize (KB)": uploaded_file.size / 1024,
    }
    st.json(file_details)

    # Read and display content if it's a text-based file
    try:
        if uploaded_file.type == "text/csv":
            import pandas as pd

            df = pd.read_csv(uploaded_file)
            st.write("### Preview of Uploaded CSV File:")
            st.dataframe(df)
        elif uploaded_file.type == "application/json":
            import json

            content = json.load(uploaded_file)
            st.write("### Content of Uploaded JSON File:")
            st.json(content)
        elif uploaded_file.type == "text/plain":
            content = uploaded_file.read().decode("utf-8")
            st.write("### Content of Uploaded Text File:")
            st.text(content)
        elif "spreadsheet" in uploaded_file.type:
            import pandas as pd

            df = pd.read_excel(uploaded_file)
            st.write("### Preview of Uploaded Excel File:")
            st.dataframe(df)
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")    
else:
    st.write("Please upload a CSV or PDF file to proceed.")
