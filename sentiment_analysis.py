import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
groq_api_key = os.environ["GROQ_API_KEY"]

#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

#chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key)


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

class Classification(BaseModel):
    sentiment: str = Field(description="The sentiment of the text")
    political_tendency: str = Field(
        description="The political tendency of the user"
    )
    language: str = Field(description="The language the text is written in")


# LLM
llm = llm.with_structured_output(
    Classification
)

tagging_chain = tagging_prompt | llm

modi_follower = "Narendra Modi’s leadership has truly transformed India. His policies have brought significant economic growth and international recognition. Proud to support him!"

rahul_follower = "Rahul Gandhi’s focus on social justice and inclusivity is commendable, but his inconsistent leadership and lack of clear policy direction often leave supporters frustrated."

response = tagging_chain.invoke({"input": modi_follower})

print("\n----------\n")

print("Sentiment analysis Modi follower:")

print("\n----------\n")
print(response)

print("\n----------\n")

response = tagging_chain.invoke({"input": rahul_follower})

print("\n----------\n")

print("Sentiment analysis Rahul follower:")

print("\n----------\n")
print(response)

print("\n----------\n")

class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    political_tendency: str = Field(
        ...,
        description="The political tendency of the user",
        enum=["conservative", "liberal", "independent"],
    )
    language: str = Field(
        ..., enum=["spanish", "english"]
    )
    
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)

# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125").with_structured_output(
#     Classification
# )

tagging_chain = tagging_prompt | llm

response = tagging_chain.invoke({"input": modi_follower})

print("\n----------\n")

print("Sentiment analysis modi follower (with a list of options using enums):")

print("\n----------\n")
print(response)

print("\n----------\n")

response = tagging_chain.invoke({"input": rahul_follower})

print("\n----------\n")

print("Sentiment analysis rahul follower (with a list of options using enums):")

print("\n----------\n")
print(response)

print("\n----------\n")
