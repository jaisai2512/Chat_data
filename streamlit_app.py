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
import streamlit.components.v1 as components

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
    o_summary,summary= summary_gen(df)
    class Message:
        """Class for keeping track of a chat message."""
        def __init__(self, origin: Literal["human", "ai"], message: str):
            self.origin = origin
            self.message = message
    
    # Load CSS
    def load_css():
        with open("static/styles.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    
    # Initialize session state
    def initialize_session_state():
        if "history" not in st.session_state:
            st.session_state.history = []
    
    # On click callback to process the message and generate a response
    def on_click_callback():
        with get_openai_callback() as cb:
            human_prompt = st.session_state.human_prompt
            message = [
                {
                    "role": "system",
                    "content": "You are an expert query interpreter for data analysis. Your job is to assist users by identifying variables involved in their query and classifying the type of output they expect.\n\n### Your Tasks:\n1. **Variable Identification**:\n   - Analyze the user's query.\n   - Identify the variables in the query that match the given list and then output the variables with the name as in the list.\n\n2. **Output Type Classification**:\n   - Determine whether the query's output is:\n     - **Visual**: If the query asks for patterns, trends, comparisons, or insights requiring a graph, chart, or plot.\n     - **Numerical**: If the query asks for specific metrics, values, or summaries.\n\n### Input Format:\nYou will be given:\n- **Query**: A user-generated question or request.\n- **Variables**: A list of possible variables, e.g., `[\"variable_1\", \"variable_2\", ..., \"variable_n\"]`.\n\n### Response Format:\nProvide your response in the following structured JSON format:\n```json\n{\n  \"matched_variables\": [\"variable_1\", \"variable_2\"],\n  \"output_type\": \"visual\" // or \"numerical\"\n}\n```\n\n### Guidelines:\n- Select the variable name from the given list that is explicitly or implicitly mentioned in the user's query, ensuring there are no missing letters in the word.\n- Use context and intent from the query to decide the output type accurately.\n- If no variable matches, leave the \"matched_variables\" array empty."
                },
                {
                    "role": "user",
                    "content": f"Query:{human_prompt}\nVariables: {df.columns}\n Please output only the json nothing apart from it"
                }
            ]
            answer = json.loads(api(message))
            var_prop = []
            for i in answer['matched_variables']:
                for j in o_summary:
                    if(j['column'] == i):
                        var_prop.append(j)
                        break
            message1 = [
                {
                    "role": "system",
                    "content": "You are a problem-solving assistant tasked with generating code in a step-by-step Chain of Thought (CoT) manner. Break down the code generation process into logical steps, ensuring each step is clear and contributes to the solution. Do not jump directly to the solution; instead, explain each step briefly while generating the corresponding code for that step. You will be given a query and the properties of the variables in the DataFrame (such as data type, missing values, etc.). The DataFrame itself is stored as 'df'. You should analyze the structure of the DataFrame based on the variable properties provided and use 'df' accordingly. If data preprocessing is needed, you should perform it as part of the process."
                },
                {
                    "role": "user",
                    "content": f"Please solve the following problem step-by-step:\n\nProblem: {human_prompt}\n\nThe variable properties (e.g., data type, missing values, etc.) will be provided."
                }
            ]
            captured_output = io.StringIO()
            sys.stdout = captured_output  # Redirect stdout
            exec(api(message1), {"df": df})  # Dangerous: Ensure you sanitize this!
            sys.stdout = sys.__stdout__  # Restore stdout
            llm_response = captured_output.getvalue().strip()
            st.session_state.history.append(Message("human", human_prompt))
            st.session_state.history.append(Message("ai", llm_response))
    
    # Main App
    load_css()
    initialize_session_state()
    
    st.title("Question Bot 🤖")
    
    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")
    credit_card_placeholder = st.empty()
    
    with chat_placeholder:
        for chat in st.session_state.history:
            div = f"""
    <div class="chat-row {'' if chat.origin == 'ai' else 'row-reverse'}">
        <img class="chat-icon" src="app/static/{'ai_icon.png' if chat.origin == 'ai' else 'user_icon.png'}" width=32 height=32>
        <div class="chat-bubble {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
            &#8203;{chat.message}
        </div>
    </div>
    """
            st.markdown(div, unsafe_allow_html=True)
    
    with prompt_placeholder:
        st.markdown("**Chat**")
        cols = st.columns((6, 1))
        cols[0].text_input("Chat", value="Hello bot", label_visibility="collapsed", key="human_prompt")
        cols[1].form_submit_button("Submit", type="primary", on_click=on_click_callback)
    
    credit_card_placeholder.caption(f"Used tokens\nDebug Langchain conversation:")
    components.html("""
    <script>
    const streamlitDoc = window.parent.document;
    const buttons = Array.from(streamlitDoc.querySelectorAll('.stButton > button'));
    const submitButton = buttons.find(el => el.innerText === 'Submit');
    streamlitDoc.addEventListener('keydown', function(e) {
        if (e.key === 'Enter') {
            submitButton.click();
        }
    });
    </script>
    """, height=0, width=0)
else:
    st.write("Please upload a CSV or PDF file to proceed.")
