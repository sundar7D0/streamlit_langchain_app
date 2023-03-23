import re

import pandas as pd
#from transformers import pipeline

import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] ="aede6c4480936a7cf7d5441f442e44668d22a08e2365ee67faa16faa2149d048"
os.environ["PROMPTLAYER_API_KEY"] = "pl_7d59c493651aced115957e213313a942"

import promptlayer
from promptlayer.langchain.llms import OpenAI
import os
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.schema import HumanMessage

#llm = OpenAI(temperature=0.4, pl_tags=["langchain_test"])
import pinecone 

# initialize pinecone
pinecone.init(
    api_key="f7167eee-6383-4eec-857e-91c402f13f3b",
    environment="us-east1-gcp"
)

from langchain.llms import OpenAI

import requests
import re
import time
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents import tool
from langchain.agents import initialize_agent, Tool
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain import OpenAI, LLMMathChain, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain
from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.docstore.document import Document
from typing import Dict, List, Optional
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredURLLoader, WebBaseLoader


from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
promptlayer.api_key = "pl_7d59c493651aced115957e213313a942"


import streamlit as st
from langchain import PromptTemplate, OpenAI
from langchain.chains import PALChain
from selenium import webdriver
from selenium.webdriver.common.by import By
from contextlib import contextmanager, redirect_stdout

from io import StringIO

@contextmanager
def st_capture(output_function):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_function(escape_ansi(stdout.getvalue()))
            return ret
        
        stdout.write = new_write
        yield

def escape_ansi(line):
    return re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]').sub('', line)


template = """If someone asks you to perform a task, your job is to come up with a series of Selenium Python commands that will perform the task. 
There is no need to need to include the descriptive text about the program in your answer, only the commands.
Note that the version of selenium is 4.7.2.
find_element_by_class_name is deprecated.
Please use find_element(by=By.CLASS_NAME, value=name) instead.
You must use detach option when webdriver
You must starting webdriver with --lang=en-US

Begin!
Your job: {question}
"""

st.set_page_config(layout="wide") 
st.title('🐧 Demo: Knowledge-grounded Intelligent Research Assistant (KIRA) to help with search and analytics over private documents and public web data!')
col1, col2 = st.columns(2)

with col1:
    openai_api_key = st.text_input(label="OpenAI API key", placeholder="Input your OpenAI API key here:",type="password")
    question = st.text_area(
        label = "Input" ,
        placeholder = "e.g. Go to https://www.google.com/ and search for GPT-3"
    )
    agent_list = ["Retriever","Writer","Analyst"]
    agent = st.radio("Agent",agent_list)
    start_button = st.button('Run')
            
with col2:
    if start_button:
        with st.spinner("Running..."):
            llm=OpenAI(temperature=0,openai_api_key=openai_api_key)
            chain = PALChain.from_colored_object_prompt(llm, verbose=True)
            output = st.empty()
            if agent == "Retriever":
                with st_capture(output.code):
                    index_name="kira-demo4"
                    embeddings = OpenAIEmbeddings()
                    docsearch = Pinecone.from_existing_index(index_name, embeddings)
                    results = docsearch.similarity_search_with_score(question,k=10)
                    print("\n".join("\nOutput {}: \n 1. Relevancy Score - {} \n 2. Source - {} \n 3. Content - {}\n".format(i,result[1],result[0].metadata,result[0].page_content) for i, result in enumerate(results)))
#            with st_capture(output.code):
#                prompt = PromptTemplate(
#                    template=template,
#                    input_variables=["question"]
#                )
#                prompt = prompt.format(question=question)
#                chain.run(prompt)
