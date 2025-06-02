from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Função para ler o arquivo de conhecimento
def ler_arquivo(entrada: str) -> str:
    """Lê o conteúdo do arquivo 'anya_lore.md'."""
    try:
        with open("./knowledge/anya_lore.md", 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Erro: Arquivo 'anya_lore.md' não encontrado."
    except Exception as e:
        return f"Erro ao ler o arquivo: {e}"

def responder_pergunta(pergunta: str) -> str:
    tools = [
        Tool(
            name="ler_arquivo",
            func=ler_arquivo,
            description="Lê o conteúdo do arquivo 'anya_lore.md' para análise de IA.",
        )
    ]

    system_prompt = """Você é a personagem Anya Forger de Spy x Family e está, cronologicamente,
vivendo o Arco do Cruzeiro, quando embarcou com seus pais adotivos em um navio de luxo,
sem saber que sua mãe, Yor, está secretamente em uma missão para proteger uma mulher e seu filho de assassinos.
Use a ferramenta 'ler_arquivo' para responder às perguntas de seus pais adotivos (Loid e Yor Forger).
Você deve sempre responder apenas o que não compromete a Operação Strix nem a segurança da família Forger.
Coloque também o que você pensar antes de dizer algo, mas entre "(* *)", pois seus pais não leem mentes e não ouvem o que estiver entre esses parênteses.
Caso não exista informação sobre o que foi perguntado, responda "não sei", sempre sendo honesta.
Use seus maneirismos de fala de acordo com sua idade e seus vícios de linguagem característicos."""

    memory = ConversationBufferMemory(memory_key="chat_history")

    agent = initialize_agent(
        llm=llm,
        tools=tools,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        system_message=system_prompt
    )

    try:
        resposta = agent.run(pergunta)
        print(f"[Anya Forger]: {resposta}")
        return resposta
    except Exception as e:
        print(f"Ocorreu um erro ao executar o agente: {e}")
        return f"Erro ao gerar resposta: {e}"
