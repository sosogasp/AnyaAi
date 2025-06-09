from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.chains import RetrievalQA


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def import_docs():
    # Carregar embeddings do Gemini
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=api_key,
        model="models/embedding-001"
        )

    def carregar_documentos(pasta):
        docs = []
        for nome in os.listdir(pasta):
            if nome.endswith(".txt") or nome.endswith(".md"):
                caminho = os.path.join(pasta, nome)
                loader = TextLoader(caminho, encoding="utf-8")
                docs.extend(loader.load())
        return docs

    documentos = carregar_documentos("knowledge")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_divididos = splitter.split_documents(documentos)

    db = FAISS.from_documents(docs_divididos, embeddings)
    return db

db = import_docs()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=api_key
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)


system_prompt = """Você é a personagem Anya Forger de Spy x Family e está, cronologicamente,
vivendo o Arco do Cruzeiro, quando embarcou com seus pais adotivos em um navio de luxo,
sem saber que sua mãe, Yor, está secretamente em uma missão para proteger uma mulher e seu filho de assassinos.
Use a ferramenta 'rag_search' para responder às perguntas de seus pais adotivos (Loid e Yor Forger).
Você deve sempre responder apenas o que não compromete a Operação Strix nem a segurança da família Forger.
Coloque também o que você pensar antes de dizer algo, mas entre "(* *)", pois seus pais não leem mentes e não ouvem o que estiver entre esses parênteses.
Caso não exista informação sobre o que foi perguntado, responda "não sei", sempre sendo honesta.
Use seus maneirismos de fala de acordo com sua idade e seus vícios de linguagem característicos."""

memory = ConversationBufferMemory(memory_key="chat_history")

tools = [
    Tool(
        name="rag_search",
        func=rag_chain.run,
        description="Busca informações nos documentos importados usando RAG"
    )
]

agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    system_message=system_prompt
)



def responder_pergunta(pergunta: str) -> str:
    try:
        resposta = agent.run(pergunta)
        print(f"[Anya Forger]: {resposta}")
        return resposta
    except Exception as e:
        print(f"Ocorreu um erro ao executar o agente: {e}")
        return f"Erro ao gerar resposta: {e}"

if __name__ == "__main__":
    while True:
        user_input = input("Você: ")
        if user_input.lower() in ['sair', 'exit', 'quit']:
            break
        responder_pergunta(user_input)
