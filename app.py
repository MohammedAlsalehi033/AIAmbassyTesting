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


#This is a test as well


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

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=5)
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

def generate_response(message):
    context = retrieve_info(message)
    response = chain.run(message=message, context=context)
    return response



app.add_middleware(
    CORSMiddleware, allow_origins=['https://yemenembassy.pk'],
    allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

@app.post("/query/")
def query_passport_service(query: QueryModel):
    try:
        response = generate_response(query.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
