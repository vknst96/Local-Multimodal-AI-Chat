from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils import load_config
import chromadb

config = load_config()

def get_ollama_embeddings():
    return OllamaEmbeddings(model=config["ollama"]["embedding_model"], base_url=config["ollama"]["base_url"])

def load_vectordb(embeddings=get_ollama_embeddings()) -> Chroma:
    """
    Loads the vector database with the specified embeddings.

    Args:
        embeddings (OllamaEmbeddings, optional): The embedding function to use. Defaults to the result of get_ollama_embeddings().

    Returns:
        Chroma: An instance of the Chroma class configured with the specified embeddings and persistent client.
    """
    persistent_client = chromadb.PersistentClient(config["chromadb"]["chromadb_path"])

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=config["chromadb"]["collection_name"],
        embedding_function=embeddings,
    )

    return langchain_chroma