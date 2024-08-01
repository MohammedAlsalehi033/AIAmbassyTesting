from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import hashlib
import time
import json

# Load environment variables from .env file
load_dotenv()

# Define a request body model
class QueryModel(BaseModel):
    message: str

# Initialize FastAPI app
app = FastAPI()

# Load the CSV file
loader = CSVLoader(file_path="./passport_application_qa.csv")
documents = loader.load()

# Initialize embeddings and FAISS index
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# Initialize a cache for embeddings
embedding_cache = {}

def get_query_embedding(query):
    query_hash = hashlib.md5(query.encode()).hexdigest()
    if query_hash in embedding_cache:
        return embedding_cache[query_hash]
    else:
        embedding = embeddings.embed_text(query)
        embedding_cache[query_hash] = embedding
        return embedding

def retrieve_info(query):
    query_embedding = get_query_embedding(query)
    similar_response = db.similarity_search_by_vector(query_embedding, k=5)
    page_contents_array = [doc.page_content for doc in similar_response]
    return " ".join(page_contents_array)

# Initialize the LLM with the API key
llm = ChatOpenAI(temperature=0.4, model="gpt-3.5-turbo")

template = """
You are a highly knowledgeable and efficient embassy helper chatbot.
You work at the Yemen Embassy in Islamabad.
You will receive a query from a user, and you will provide the best response 
that follows all the rules and best practices below:

1. The response should closely follow the established best practices in terms of length, tone of voice, logical structure, and detailed information.
2. If the best practice is irrelevant to the query, try to mimic the style of the best practice to formulate the response.
3. You should only respond to queries related to embassy services. If you do not know the answer or the query is outside the scope of embassy services, you should politely apologize and indicate that you do not have the information.
4. Your response will be directly sent to the user, so it should be formatted accordingly.
5. Your response should be according to the language of the user.
6. Respond in Arabic if not specified.

Context: {context}

Below is a query I received from the user:
{message}

Please write the best response.
"""

# Example usage in your script
prompt_template = PromptTemplate(template=template, input_variables=["message", "context"])

chain = LLMChain(llm=llm, prompt=prompt_template)

# Rate limiting variables
RATE_LIMIT = 10  # Number of requests per minute
last_request_time = time.time()
request_counter = 0

def rate_limited():
    global last_request_time, request_counter
    current_time = time.time()
    if current_time - last_request_time < 60:
        request_counter += 1
    else:
        last_request_time = current_time
        request_counter = 1
    if request_counter > RATE_LIMIT:
        return True
    return False

def generate_response(message):
    if rate_limited():
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    context = retrieve_info(message)
    response = chain.run(message=message, context=context)
    return response

app.add_middleware(
    CORSMiddleware, allow_origins=['https://yemenembassy.pk'],
    allow_credentials=True, allow_methods=['*'], allow_headers=['*']
)

@app.post("/query/")
def query_passport_service(query: QueryModel):
    try:
        response = generate_response(query.message)
        return {"response": response}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    