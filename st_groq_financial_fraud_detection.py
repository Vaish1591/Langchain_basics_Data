import streamlit as st
import os
from langchain_core.prompts import PromptTemplate
import warnings
from langchain._api import LangChainDeprecationWarning
from langchain_core.messages import HumanMessage
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
groq_api_key = os.environ["GROQ_API_KEY"]
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from langchain.docstore.document import Document
import pandas as pd

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
# Ensure NLTK resources are downloaded
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

st.set_page_config(
    page_title="Financial fraud detection"
)
st.title("Evaluate a RAG App for financial frauds")

with st.expander("Evaluate a RAG App for financial frauds"):
    st.write("""
        Input: Financial statements and related documents.

        Process:The system uses RAG to retrieve pertinent information from a database and employs LLM to analyze and interpret the data.

        Output:A concise report indicating whether the financial statement exhibits fraudulent behavior, with an explanation based on the retrieved context.

        This combination of LLM and RAG enhances the accuracy and reliability of fraud detection in financial filings, making it a powerful tool for auditors, regulators, and financial institutions.
    """)

uploaded_file = st.file_uploader(
    "Upload a input csv document",
    type="csv"
)

def clean_text(text):
    # Remove non-ASCII characters
    text = text.encode('ascii', 'ignore').decode()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into text
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def process_data(df):
    df['Clean_Text'] = df['text'].apply(clean_text)

    # Drop original 'Text' column if no longer needed
    df.drop(columns=['text'], inplace=True)
    return df


if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    processed_df = process_data(df)
    
    documents = []

    # Iterate over rows using .rows() method
    for i, row_tuple in processed_df.iterrows():
        document = f"id:{i}\Fillings: {row_tuple[1]}\Fraud_Status: {row_tuple[0]}"
        documents.append(Document(page_content=document))

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # create a vectorstore and store there the texts
    db = FAISS.from_documents(documents, embeddings)
    # create a retriever interface
    retriever = db.as_retriever()
    
    prompt_template = """
    You are a financial expert. Given the following financial statement, determine if it is fraudulent or not. 
    Provide a detailed explanation for your decision.
    
    Financial Statement:
    {query}
    Context: {context}
    
    Is this statement fraudulent? (yes/no):
    Answer:
    """
    
    prompt = PromptTemplate(
        input_variables=["query"],
        template=prompt_template
    )

    # Regular QA chain
    qachain = RetrievalQA.from_chain_type(
        llm=ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key),
        chain_type="stuff",
        retriever=retriever
    )
    
    # User query input
    query_text = st.text_input(
        "Enter the financial statement you want to check for fraud:",
        placeholder="Write the financial statement here"
    )
    
    if query_text:
        # Predictions
        prediction = qachain.run({"query": query_text})
        
        st.write("Financial Statement:")
        st.info(query_text)
        st.write("Fraud Detection Result:")
        st.info(prediction)
    


