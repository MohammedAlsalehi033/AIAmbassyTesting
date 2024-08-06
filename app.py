from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
import pickle
import faiss



# Load environment variables from .env file
load_dotenv()

# Define a request body model
class QueryModel(BaseModel):
    message: str

# Initialize FastAPI app
app = FastAPI()




app.add_middleware(
    CORSMiddleware, 
    allow_origins=['https://yemenembassy.pk'],
    allow_credentials=True, 
    allow_methods=['*'], 
    allow_headers=['*']
)



loader = CSVLoader(file_path="./passport_application_qa.csv")
documents = loader.load()

model = SentenceTransformer('all-MiniLM-L6-v2')

document_texts = [doc.page_content for doc in documents]

document_embeddings = model.encode(document_texts)

with open('./documents.pkl', 'wb') as f:
    pickle.dump(document_texts, f)

with open('./embeddings.pkl', 'wb') as f:
    pickle.dump(document_embeddings, f)

d = document_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(document_embeddings)

faiss.write_index(index, './faiss_index.index')

def retrieve_info(query):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=5)
    results = [document_texts[i] for i in I[0]]
    return " ".join(results)

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

def generate_response(message):
    context = retrieve_info(message)
    response = chain.run(message=message, context=context)
    return response



@app.post("/query/")
def query_passport_service(query: QueryModel):
    try:
        response = generate_response(query.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
