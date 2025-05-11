from langchain_community.llms import HuggingFaceHub
import os

# Define the repo ID and connect to Mixtral model on Huggingface

def get_model():
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
    repo_id=repo_id, 
    model_kwargs={"temperature": 0.8, "top_k": 50}, 
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )
    return llm