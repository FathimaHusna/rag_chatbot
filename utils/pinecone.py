from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from utils.get_embeddings import get_embeddings
import os

# Initialize Pinecone client

def get_pinecone():
    """
    Initialize Pinecone client and create or link to an index.
    """
    # Initialize Pinecone
    pc = Pinecone(
        api_key= os.getenv('PINECONE_API_KEY')
    )

    # Define Index Name
    index_name = "langchain-demo"
    

    docs, embeddings = get_embeddings()

    # Checking Index
    # if index_name not in Pinecone.list_indexes():
    # Create new Index
    # pc.create_index(
    #     name=index_name,
    #     dimension=1024, # Replace with your model dimensions
    #     metric="cosine", # Replace with your model metric
    #     spec=ServerlessSpec(
    #         cloud="aws",
    #         region="us-east-1"
    #     ) 
    # )
    # docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
    docsearch = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=['How is my luck in 2025!'],
                    parameters={
                        "input_type": "query"
                    }
    )
    print('docsearch = ', docsearch)
    # else:
    # # Link to the existing index
    #     docsearch = Pinecone.from_existing_index(index_name, embeddings)
    
    return docsearch
