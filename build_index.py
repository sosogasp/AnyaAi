from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings  # Ou o embedding que for usar

def build_and_save_index():
    # Caminho do arquivo na raiz
    loader = TextLoader("knowledge/anya_lore.md", encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()  # Troque se usar outro embedding
    vectordb = Chroma.from_documents(docs_split, embeddings, persist_directory="chroma_index")
    vectordb.persist()

if __name__ == "__main__":
    build_and_save_index()
