import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the .env file

# Load the CSV file
loader = CSVLoader(file_path="./passport_application_qa.csv")
documents = loader.load()

# Initialize embeddings with the API key
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    return page_contents_array

# Initialize the LLM with the API key
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

template = """
You are a highly knowledgeable and efficient embassy helper chatbot.
I will share a query from a user with you, and you will provide the best response 
that follows all the rules and best practices below:

1/ The response should closely follow the established best practices in terms of length, tone of voice, logical structure, and detailed information.

2/ If the best practice is irrelevant to the query, try to mimic the style of the best practice to formulate the response.

3/ You should only respond to queries related to embassy services. If you do not know the answer or the query is outside the scope of embassy services, you should politely apologize and indicate that you do not have the information.

Below is a query I received from the user:
{message}

Here is a list of best practices of how we normally respond to users in similar scenarios:
{best_practice}

Please write the best response that I should send to this user:
"""

# Example usage in your script
prompt_template = PromptTemplate(template=template, input_variables=["message", "best_practice"])

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response

def main():
    st.set_page_config(
        page_title="Digital ambassador", page_icon="📄")

    st.header("Digital ambassador 📄")
    message = st.text_area("customer message")

    if message:
        st.write("Generating Response...")

        result = generate_response(message)

        st.info(result)

if __name__ == '__main__':
    main()
