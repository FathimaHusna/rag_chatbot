from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """
    Load and split documents, then create embeddings.
    """
    # Load and split the document
    loader = TextLoader('./horoscope.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    return docs, embeddings