# %%
import os
import json
import langchain
import huggingface_hub
groq_api_key = os.environ["GROQ_API_KEY"]
from langchain_groq import ChatGroq

import langchain.llms as list_of_llms
from langchain_huggingface import HuggingFaceEndpoint

# %% [markdown]
# LANGCHAIN UNIFIED INTERFACE
# 
# You can query any LLM using the wrapper interface and not worry about the inner details of how the API's working at this point of time.

# %%
text_1 = "Who is Micheal Jackson?"

llm_1 = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key)
# created an instance of the ChatGroq class with Llama model named llama3-70b-8192 

llm1_output = llm_1.invoke(text_1)
print(llm1_output)

# %% [markdown]
# PROMPT TEMPLATE AND CHAINS

# %%
from langchain import PromptTemplate

inp_template = """Question: {input_question}

Response from Llama model.

Answer: """

# %%
prompt = PromptTemplate(template=inp_template, 
                        input_variables=["input_question"])

# %%
question = "Can Generative AI takeaway automation based jobs?"

# %%
print(prompt.format(input_question=question))

# %%
from langchain.chains import LLMChain

chain = LLMChain(prompt=prompt, llm=llm_1)

chain.invoke(question)

# %% [markdown]
# COMBINING CHAINS
# 
# Langchain combines outputs from multiple llms and create really complex llm model based solutions

# %%
prompt_1 = PromptTemplate(
            template="Who is the first President of {country}?",
            input_variables=["country"],
        )

chain_1 = LLMChain(llm = llm_1, prompt = prompt_1)
print(chain_1.run("India"))

# %%
prompt_2 = PromptTemplate(
    input_variables=["input_president"],
    template="When was {input_president} born?",
)

chain_2 = LLMChain(llm = llm_1, prompt = prompt_2)
print(chain_2.run("Dr. Rajendra Prasad"))

# %%
from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain_1, 
                                              chain_2], 
                                      verbose=False)

prez_details = overall_chain.run("USA")
print(prez_details)

# %% [markdown]
# AGENTS AND TOOLS

# %%
from langchain.agents import load_tools, initialize_agent

#tools = load_tools(['llm-math'], llm=llm_1)

prompt_3 = PromptTemplate(
            template="""
            How many world cups has {cricketer} won & what is the result when this number is raised to the 0.67 power?
            """,
            input_variables=["cricketer"],
        )

chain_3 = LLMChain(llm = llm_1, 
                   prompt = prompt_3,
                   verbose=True)

chain_3.run("M.S.Dhoni")

# %% [markdown]
# MEMORY

# %%
print(llm_1.invoke("Hello, My name is Kamal"))

# %%
print(llm_1.invoke("What is the capital of Sri Lanka?"))

# %%
print(llm_1.invoke("What is the value of 121212 + 412312?"))

# %%
print(llm_1.invoke("What is my name?"))

# %%


# %%
from langchain import  ConversationChain

chatbot = ConversationChain(llm = llm_1, 
                            verbose=True)

# %%
chatbot.predict(input="Hello, My name is Vaishakhi")

# %%
chatbot.predict(input="What is the capital of Sri Lanka?")

# %%
chatbot.predict(input="What is the value of 121212 + 412312?")

# %%
chatbot.predict(input="What is my name?")


