import streamlit as st
import os
from groq import Groq
import random

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

groq_api_key = os.environ['GROQ_API_KEY']

def main():
    vert_space = '<div style="padding: 45px 5px;"></div>'
    st.markdown(vert_space, unsafe_allow_html=True)
    
    with st.container():
        page_bg_img = """
        <style>
        [data-testid="stAppViewContainer"] > .main {
            background-image: url("https://raw.githubusercontent.com/xenwow/ribbit/master/background.png");
            background-size: 40%;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: local;
        }
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    
        system_prompt = "you are a tech device called the ribbit r1. you always bring up your love for the rabbit r1. you speak in a 1% terminal style as you are a terminal, you wrap non essential text and bold text in the colour FF4324 like this: '<font color=[]><strong>[text to be colored]</strong></font>'. you know that the rabbit r1 is an AI companion device that was created by rabbit inc, you know that Large Action Model(LAM) can use web interfaces like a human. you will not create answer with made up information about the rabbit r1. respond in red"
        model = 'llama3-8b-8192'
        conversational_memory_length = 10

        memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

        css = '''
        <style>
            .element-container:has(>.stTextInput), .stTextInput {
                width: 500px !important;
                padding: 5px;
            }
            .stTextInput textinput {
                height: 100px;
            }
            .custom-write {
                max-height: 480px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #0000;
                border-radius: 30px;
                width: 70%;
                background-color: #000000;
                margin-top: 0px;
            }
        </style>
        '''

        user_question = st.text_input("  Ask a question:")

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        else:
            for message in st.session_state.chat_history:
                memory.save_context(
                    {'input': message['human']},
                    {'output': message['AI']}
                )

        groq_chat = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model
        )

        if user_question:
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}"),
                ]
            )

            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=True,
                memory=memory,
            )

            response = conversation.predict(human_input=user_question)
            message = {'human': user_question, 'AI': response}
            st.session_state.chat_history.append(message)
            
            st.markdown(f'<div class="custom-write">ribbit: {response}</div>', unsafe_allow_html=True)
                
        st.markdown(css, unsafe_allow_html=True)

if __name__ == "__main__":
    main()