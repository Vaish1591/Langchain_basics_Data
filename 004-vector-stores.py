import os
import nltk
nltk.download('averaged_perceptron_tagger')
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
groq_api_key = os.environ["GROQ_API_KEY"]

#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

#chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")
chatModel = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key)

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
loaded_document = TextLoader("./data/be-good.txt").load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

chunks_of_text = text_splitter.split_documents(loaded_document)

vector_db = Chroma.from_documents(chunks_of_text, HuggingFaceEmbeddings())

question = "What was the name mentioned in the conversation?"

response = vector_db.similarity_search(question)

print("\n----------\n")

print("Ask the RAG App: What was the name mentioned in the conversation?")

print("\n----------\n")
print(response[0].page_content)

print("\n----------\n")

