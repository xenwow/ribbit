import streamlit as st
import os
from groq import Groq
import random
import base64
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os 

load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']



def main():

   
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://raw.githubusercontent.com/xenwow/ribbit/master/background.png");
    background-size: 80%;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: local;
    }}
    </style>
    """

    st.markdown(page_bg_img, unsafe_allow_html=True)
  
    
   
    conversational_memory_length = 10

    memory=ConversationBufferWindowMemory(k=conversational_memory_length)

    user_question = st.text_area("Ask a question:")

    # session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    else:
        for message in st.session_state.chat_history:
            memory.save_context({'input':message['user']},{'output':message['AI']})


    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama3-8b-8192"
    )

    conversation = ConversationChain(
            llm=groq_chat,
            memory=memory
    )

    if user_question:
        response = conversation(user_question)
        message = {'user':user_question,'AI':response['response']}
        st.session_state.chat_history.append(message)
        st.write("ribbit:", response['response'])

if __name__ == "__main__":
    main()