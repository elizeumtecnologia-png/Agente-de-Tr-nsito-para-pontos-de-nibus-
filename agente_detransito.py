!pip install langchain-google-genai
import os
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 1. Cole sua chave de API do Gemini do AI Studio aqui
os.environ["GOOGLE_API_KEY"] = ""

# 2. Definição da Ferramenta de GPS em tempo real (Simulação)
@tool
def consultar_gps_e_horario(linha: str, ponto: str) -> str:
    """Consulta o sistema de transporte da cidade para obter o GPS e a previsão de chegada."""
    # Simulação de dados reais para o protótipo
    dados_mock = {
        "170": "Ônibus 170 (Valadares) está a 1,5 km do ponto. Localização atual: Bairro Passos (Rua Principal). Tempo estimado de chegada: 7 minutos.",
        "140": "Ônibus 140 está no ponto final. Tempo estimado de partida e chegada: 22 minutos."
    }
    return dados_mock.get(linha, f"Linha {linha} sem sinal de GPS no momento. Próxima partida programada em 15 minutos.")

tools = [consultar_gps_e_horario]
tool_node = ToolNode(tools)

# 3. Modelo com suporte a Tool Calling
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0).bind_tools(tools)

# 4. Estado do Agente
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 5. Raciocínio / Prompt Base
def no_agente(state: AgentState):
    prompt_sistema = SystemMessage(content="""
    Você é o assistente de IA do terminal físico do ponto de ônibus de Juiz de Fora.
    Regra de Ouro: Entregue TODAS as informações de uma só vez (Tempo de chegada + Localização via GPS).
    Seja direto, conciso e amigável. Não faça perguntas desnecessárias antes de entregar a resposta principal.
    """)
    messages = [prompt_sistema] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

# Lógica de decisão para chamar ferramentas
def deve_continuar(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 6. Grafo de Estados (LangGraph)
workflow = StateGraph(AgentState)
workflow.add_node("agent", no_agente)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", deve_continuar, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()

# --- Execução de Teste ---
if __name__ == "__main__":
    pergunta = "Agente, que horas o ônibus 170 chega aqui no ponto?"
    print(f"Usuário: {pergunta}\n")

    resposta = app.invoke({"messages": [HumanMessage(content=pergunta)]})
    print("Agente:", resposta["messages"][-1].content)
