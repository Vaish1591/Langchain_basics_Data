import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
import warnings
from langchain._api import LangChainDeprecationWarning
from langchain_core.messages import HumanMessage
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
groq_api_key = os.environ["GROQ_API_KEY"]
from langchain_groq import ChatGroq

import os

st.set_page_config(
    page_title = "Blog Post Generator"
)

st.title("Blog Post Generator")

def generate_response(topic_text):
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key)
    template = """
    As experienced startup and venture capital writer, 
    generate a 400-word blog post about {topic_text}
    
    Your response should be in this format:
    First, print the blog post.
    Then, sum the total number of words on it and print the result like this: This post has X words.
    """
    prompt = PromptTemplate(
        input_variables = ["topic_text"],
        template = template
    )
    formatted_prompt = prompt.format(
        topic_text=topic_text
    )
    improved_redaction = llm.invoke([HumanMessage(content=formatted_prompt)])
    return st.write(improved_redaction)

topic_text = st.text_input("Enter topic: ")
generate_response(topic_text)
