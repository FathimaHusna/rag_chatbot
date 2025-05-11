import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# SECTION 1: Imports
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Pinecone as PineconeVectorStore # Langchain's Pinecone wrapper
from langchain_huggingface import HuggingFaceEndpoint
from pinecone import Pinecone as PineconeClient, ServerlessSpec # Pinecone SDK client
import pinecone # Importing to access __version__

# SECTION 2: Configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = "ragbot" # Ensure this matches your desired index name
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

EMBEDDING_MODEL_NAME = "sentence-transformers/all-roberta-large-v1" # Uses default for HuggingFaceEmbeddings ('sentence-transformers/all-MiniLM-L6-v2')

# --- IMPORTANT ---
# Set to True for the first run to create the index and ingest documents.
# Set to False for subsequent runs to connect to the existing index.
INGEST_DOCUMENTS = True # Change as needed
# --- ----------- ---

# SECTION 3: Utility Functions

def get_embedding_function():
    """
    Initializes and returns the HuggingFace embedding function.
    """
    if EMBEDDING_MODEL_NAME:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    # LangChainDeprecationWarning: Default values for HuggingFaceEmbeddings.model_name were deprecated
    # Explicitly pass a model_name. Using default 'sentence-transformers/all-MiniLM-L6-v2' for now.
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def load_and_split_documents(file_path='./horoscope.txt'):
    """
    Load and split documents from a text file.
    Creates a dummy file if it doesn't exist and INGEST_DOCUMENTS is True.
    """
    if not os.path.exists(file_path) and INGEST_DOCUMENTS:
        print(f"'{file_path}' not found. Creating a dummy file for ingestion.")
        with open(file_path, 'w', encoding='utf-8') as f: # Added encoding
            f.write("Aries: Today is a good day for new beginnings. Trust your intuition.\n")
            f.write("Taurus: Financial matters might need your attention. Be practical.\n")
            f.write("Gemini: Communication is key today. Express yourself clearly.\n")
            f.write("Cancer: Focus on home and family. Emotional connections are important.\n")
            f.write("Leo: Creativity and self-expression are highlighted. Shine brightly!\n")
            f.write("Virgo: Pay attention to details, especially in your work. Organization helps.\n")
            f.write("Libra: Seek balance in relationships and decisions. Harmony is achievable.\n")
            f.write("Scorpio: Transformation and deep insights are possible. Embrace change.\n")
            f.write("Sagittarius: Adventure calls! Broaden your horizons and learn something new.\n")
            f.write("Capricorn: Discipline and hard work will pay off. Stay focused on your goals.\n")
            f.write("Aquarius: Innovations and new ideas may come to you. Think outside the box.\n")
            f.write("Pisces: Your intuition is strong. Listen to your inner voice and dreams.\n")
    elif not os.path.exists(file_path):
        print(f"Warning: Document file '{file_path}' not found.")
        return []

    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    return docs

_pinecone_client = None

def get_pinecone_sdk_client():
    global _pinecone_client
    if _pinecone_client is None:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable not set.")
        _pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)
    return _pinecone_client

def get_pinecone_vector_store():
    """
    Initializes and returns a Langchain Pinecone vector store.
    Handles index creation and population based on INGEST_DOCUMENTS flag.
    """
    pc_sdk_client = get_pinecone_sdk_client()
    embedding_func = get_embedding_function()

    try:
        test_embedding = embedding_func.embed_query("test")
        embedding_dimension = len(test_embedding)
        print(f"Detected embedding dimension: {embedding_dimension} (from {EMBEDDING_MODEL_NAME or 'default HF model'})")
    except Exception as e:
        print(f"Error determining embedding dimension: {e}. Please ensure your embedding model is working.")
        raise

    # --- Updated Index Existence Check for Pinecone SDK v6.x.x ---
    index_exists = False
    print(f"DEBUG: Pinecone SDK Client type: {type(pc_sdk_client)}")
    print(f"DEBUG: Attempting to list Pinecone indexes using SDK v{pinecone.__version__}")

    try:
        list_indexes_response = pc_sdk_client.list_indexes()
        print(f"DEBUG: Type of list_indexes_response: {type(list_indexes_response)}")
        # print(f"DEBUG: Value of list_indexes_response: {list_indexes_response}") # Can be very verbose

        actual_index_names_list = []

        # Check if list_indexes_response itself is a list of Index objects (or similar with a 'name' attribute)
        # This is a common pattern for newer SDKs (e.g., pinecone v3+ IndexList.indexes)
        if hasattr(list_indexes_response, 'indexes') and isinstance(list_indexes_response.indexes, list):
            print("DEBUG: Found 'indexes' attribute on list_indexes_response. Iterating through it.")
            for index_info in list_indexes_response.indexes:
                if hasattr(index_info, 'name'):
                    actual_index_names_list.append(index_info.name)
                else:
                    print(f"DEBUG: Index info object {index_info} (type: {type(index_info)}) does not have a 'name' attribute.")
            print(f"DEBUG: Names extracted from .indexes attribute: {actual_index_names_list}")
        
        # Fallback or alternative: Check if list_indexes_response has a 'names' attribute/method
        # This was more common with the IndexList object in v3/v4
        elif hasattr(list_indexes_response, 'names'):
            names_member = getattr(list_indexes_response, 'names')
            print(f"DEBUG: Found 'names' member on list_indexes_response. Type: {type(names_member)}")
            if callable(names_member):
                print("DEBUG: 'names' member is callable. Calling it.")
                actual_index_names_list = names_member() # Call the method
            elif isinstance(names_member, list):
                print("DEBUG: 'names' member is already a list.")
                actual_index_names_list = names_member
            else:
                print(f"DEBUG: 'names' member (type: {type(names_member)}) is neither callable nor a list. Trying to iterate.")
                try:
                    actual_index_names_list = list(names_member) # Try to convert if iterable
                    print(f"DEBUG: Successfully converted 'names' member to list: {actual_index_names_list}")
                except TypeError:
                    print(f"DEBUG: Failed to convert 'names' member to list. It's not iterable.")
        
        # Check if list_indexes_response itself is the list of name strings (very old client behavior)
        elif isinstance(list_indexes_response, list) and all(isinstance(item, str) for item in list_indexes_response):
            print("DEBUG: list_indexes_response itself is a list of strings (old client behavior).")
            actual_index_names_list = list_indexes_response
        
        else:
            print(f"DEBUG: Could not determine how to extract index names from list_indexes_response of type {type(list_indexes_response)}. Value: {list_indexes_response}")
            # If no method worked, actual_index_names_list will remain empty or as previously set

        if not isinstance(actual_index_names_list, list):
            # This block tries to ensure actual_index_names_list is indeed a list
            print(f"WARNING: Extracted index names is not a list yet. Got type: {type(actual_index_names_list)}. Value: {actual_index_names_list}")
            if isinstance(actual_index_names_list, str): # e.g. if API returned a single index name as string
                 actual_index_names_list = [actual_index_names_list]
            else:
                # This is a fallback if no known structure was matched and it's not already a list.
                # It could happen if list_indexes_response is an unexpected type.
                print(f"ERROR: Could not obtain a list of index names. Final actual_index_names_list type: {type(actual_index_names_list)}")
                # For safety, we'll re-initialize to empty list to avoid errors later, but this indicates a problem.
                actual_index_names_list = [] # This is a fallback

        print(f"DEBUG: Final list of index names before check: {actual_index_names_list}")
        if PINECONE_INDEX_NAME in actual_index_names_list:
            index_exists = True
        print(f"DEBUG: Index '{PINECONE_INDEX_NAME}' exists: {index_exists}")

    except AttributeError as e:
        print(f"DEBUG: AttributeError during index listing (likely API change in Pinecone SDK): {e}")
        import traceback
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"DEBUG: An unexpected error occurred during Pinecone index check: {e}")
        import traceback
        traceback.print_exc()
        raise
    # --- End of Updated Index Existence Check ---

    vector_store = None # Initialize
    if not index_exists:
        if INGEST_DOCUMENTS:
            print(f"Index '{PINECONE_INDEX_NAME}' not found or list empty. Creating new index with dimension {embedding_dimension}...")
            pc_sdk_client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1" # Ensure this is correct for your Pinecone setup
                )
            )
            print(f"Index '{PINECONE_INDEX_NAME}' created. Populating with documents...")
            docs_to_ingest = load_and_split_documents()
            if not docs_to_ingest:
                print(f"Warning: No documents found to populate the index '{PINECONE_INDEX_NAME}'. The index will be empty.")
                # Connect to the newly created empty index
                vector_store = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embedding_func)
            else:
                vector_store = PineconeVectorStore.from_documents(
                    docs_to_ingest,
                    embedding_func,
                    index_name=PINECONE_INDEX_NAME
                )
            print(f"Documents populated into index '{PINECONE_INDEX_NAME}'.")
        else:
            raise ValueError(
                f"Index '{PINECONE_INDEX_NAME}' does not exist or was not found in the list, and INGEST_DOCUMENTS is False. "
                "Set INGEST_DOCUMENTS=True to create and populate it, or ensure the index exists and is accessible."
            )
    else: # Index exists
        if INGEST_DOCUMENTS:
            print(f"Index '{PINECONE_INDEX_NAME}' exists. Re-populating with documents (this will add/update documents)...")
            docs_to_ingest = load_and_split_documents()
            if not docs_to_ingest:
                 print(f"Warning: No documents found to re-populate index '{PINECONE_INDEX_NAME}'. Existing data remains.")
                 vector_store = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embedding_func)
            else:
                vector_store = PineconeVectorStore.from_documents(
                    docs_to_ingest,
                    embedding_func,
                    index_name=PINECONE_INDEX_NAME
                )
            print(f"Documents (re-)populated into index '{PINECONE_INDEX_NAME}'.")
        else:
            print(f"Connecting to existing index '{PINECONE_INDEX_NAME}'.")
            vector_store = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embedding_func)
    
    if vector_store is None:
        # This case should ideally not be reached if logic above is correct
        raise RuntimeError("Failed to initialize Pinecone vector_store.")
        
    return vector_store


def get_prompt_template():
    template = """
    You are a fortune teller. The Human will ask you a question about their life. 
    Use the following piece of context to answer the question. 
    If you don't know the answer from the context, or if the context is not relevant,
    just say you don't have a specific prediction for that from the provided horoscopes.
    Keep the answer within 2-3 sentences and concise.

    Context: {context}
    Question: {question}
    Answer: 
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    return prompt

def get_llm_model():
    if not HUGGINGFACE_API_KEY:
        raise ValueError("HUGGINGFACE_API_KEY environment variable not set.")
    
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    # Using the newer HuggingFaceEndpoint
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY,
        temperature=0.7, # Common way to set temperature
        # model_kwargs can still be used for other specific params if needed by the endpoint
        # but common ones like temperature, max_tokens are often direct args
        max_new_tokens=100, # Example: if you want to control output length
        top_k=50,
        # task="text-generation" # Often inferred, but can be explicit
    )
    return llm

# SECTION 4: Chatbot Class

class Chatbot():
    def __init__(self):
        print("Initializing Chatbot components...")
        self.docsearch = get_pinecone_vector_store()
        print("Pinecone vector store initialized.")
        self.prompt = get_prompt_template()
        print("Prompt template initialized.")
        self.llm = get_llm_model()
        print("LLM model initialized.")

        self.rag_chain = (
            {"context": self.docsearch.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        print("RAG chain built. Chatbot is ready.\n")

# SECTION 5: Main Execution
if __name__ == "__main__":
    if not PINECONE_API_KEY or not HUGGINGFACE_API_KEY:
        print("ERROR: PINECONE_API_KEY or HUGGINGFACE_API_KEY not found in environment variables or .env file.")
        print("Please ensure they are set correctly.")
        exit(1)

    print(f"--- Chatbot Configuration ---")
    print(f"Pinecone Index: {PINECONE_INDEX_NAME}")
    print(f"Pinecone SDK Version: {pinecone.__version__}")
    print(f"Ingest Documents on Start: {INGEST_DOCUMENTS}")
    if EMBEDDING_MODEL_NAME:
        print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    else:
        print(f"Embedding Model: Using default 'sentence-transformers/all-MiniLM-L6-v2'")
    print(f"--- --------------------- ---")

    try:
        bot = Chatbot()
        
        print("Welcome to the Fortune Teller Chatbot!")
        print("Type 'exit' or 'quit' to end the chat.")
        
        while True:
            user_input = input("\nAsk me anything about your fortune: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye! May your future be bright.")
                break
            if not user_input.strip():
                continue
            
            print("Thinking...")
            try:
                result = bot.rag_chain.invoke(user_input)
                print("\nFortune Teller says:")
                print(result)
            except Exception as e:
                print(f"An error occurred while getting your fortune: {e}")
                print("This might be due to API rate limits or a temporary issue with the services.")

    except Exception as e:
        print(f"\nAn critical error occurred during Chatbot initialization or execution: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the following:")
        print("1. Your .env file has correct and active API keys.")
        print("2. If INGEST_DOCUMENTS=True, 'horoscope.txt' exists or can be created.")
        print("3. If INGEST_DOCUMENTS=False, the Pinecone index exists and is populated with compatible embeddings.")
        print("4. You have a stable internet connection and necessary permissions for API access.")
        print("5. All required Python packages are installed and versions are compatible.")