from langchain_core.tools import tool
import requests

metal_data = {
    "unit":"gram",
    "currency": "USD",
    "prices": {
        "gold": 88.1553,
        "silver": 1.0523,
        "platinum": 32.169,
        "palladium": 35.8252,
        "copper": 0.0098,
        "aluminum": 0.0026,
        "lead": 0.0021,
        "nickel": 0.0159,
        "zinc": 0.0031,
    }
}


# Define the tools for the agent to use, it is necessary to specify that each function is a tool
@tool
def get_metal_price(metal_name: str) -> str:
    """Fetches the current per gram in USD price of the specified metal.

    Args:
        metal_name : The name of the metal (e.g., 'gold', 'silver', 'platinum').

    Returns:
        float: The current price of the metal in dollars per gram.

    Raises:
        KeyError: If the specified metal is not found in the data source.
    """
    try:
        metal_name = metal_name.lower().strip()
        prices = metal_data["prices"]
        currency = metal_data["currency"]
        unit=metal_data["unit"]
        if metal_name not in prices:
            raise KeyError(
                f"Metal {metal_name} not found. Available metals: {', '.join(metal_data['prices'].keys())}"
            )
        price=prices[metal_name]
        return f"The current price of {metal_name} is {price} {currency} per {unit}."
    except Exception as e:
        raise Exception(f"Error fetching metal price: {str(e)}")
     
@tool    
def get_currency_exchange(base: str, target: str) -> str:
    """
    Returns the exchange rate from base currency to target currency.

    Args:
        base (str): The base currency (e.g., 'USD').
        target (str): The target currency (e.g., 'EUR').

    Returns:
        str: A human-readable string showing the exchange rate,
             or an error message if the pair is not found.
    """
    fake_rates = {
        ("usd", "eur"): 0.8,
        ("eur", "usd"): 1.8,
        ("usd", "gbp"): 0.7
    }
    rate = fake_rates.get((base.lower(), target.lower()))
    if rate is None:
        return f"No exchange rate found for {base.upper()} to {target.upper()}"
    return  f"{base.upper()} = {rate} {target.upper()}"

@tool
def search_duckduckgo(query: str) -> str:
    """
    Performs a DuckDuckGo search using a JSON endpoint and returns the first result snippet.
    """
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if "AbstractText" in data and data["AbstractText"]:
            return data["AbstractText"]
        elif data.get("RelatedTopics"):
            for topic in data["RelatedTopics"]:
                if isinstance(topic, dict) and topic.get("Text"):
                    return topic["Text"]
        return "No relevant results found."
    except Exception as e:
        return f"Search error: {e}"


from langchain_ollama.chat_models import ChatOllama
from langgraph.prebuilt import create_react_agent





# Instanciamos el modelo LLM local usando Ollama (Llama 3.2)
llm = ChatOllama(
    model="llama3.2",   # Usamos el modelo Llama 3.2 local
    temperature=0
)

# Vinculamos nuestra herramienta al LLM
tools = [get_metal_price,get_currency_exchange]
llm_with_tools = llm.bind_tools(tools)
agent = create_react_agent(
    model=llm_with_tools,
    tools=[get_metal_price, get_currency_exchange],
    prompt="""
You are a ReAct agent. For "What is the price of METAL in CUR?":
1) Call get_metal_price with METAL.
2) Then call get_currency_exchange with base='USD', target=CUR.
3) Finally, respond combining both results.
"""
)
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

# 1) State
class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


# 2) Nodes
tools = [get_metal_price,get_currency_exchange]
tool_node = ToolNode(tools)

def assistant(state: GraphState):
    # 1) Pasar todo el historial al agente ReAct
    result = agent.invoke({"messages": state["messages"]})
    
    # 2) Extraer la lista de nuevos mensajes
    new_msgs = result["messages"]
    
    # 3) Devolver el historial completo concatenado
    return {"messages": state["messages"] + new_msgs}

def should_continue(state: GraphState):
    last = state["messages"][-1]

    # 1) Si es un dict y contiene tool_calls
    if isinstance(last, dict) and last.get("tool_calls"):
        return "tools"

    # 2) Si es un objeto con atributo tool_calls (p.ej. AIMessage)
    tc = getattr(last, "tool_calls", None)
    if tc:
        return "tools"

    return END

from langgraph.graph import START, StateGraph
from IPython.display import Image, display

# Define a new graph for the agent
builder = StateGraph(GraphState)

# Define the two nodes we will cycle between
builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)

# Set the entrypoint as `agent`
builder.add_edge(START, "assistant")

# Making a conditional edge
# should_continue will determine which node is called next.
builder.add_conditional_edges("assistant", should_continue, {"tools", END}) # type: ignore

# Making a normal edge from `tools` to `agent`.
# The `agent` node will be called after the `tool`.
builder.add_edge("tools", "assistant")

# Compile and display the graph for a visual overview
react_graph = builder.compile()
#display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))

from langchain.schema.messages import AIMessage
from copy import deepcopy
import json
import uuid

def fix_tool_calls_for_openai_format(messages):
    fixed_messages = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            new_msg = deepcopy(msg)
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                formatted_tool_calls = []
                for tool_call in tool_calls:
                    formatted = {
                        "id": tool_call.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["args"])
                        }
                    }
                    formatted_tool_calls.append(formatted)
                    print(formatted)
                new_msg.additional_kwargs["tool_calls"] = formatted_tool_calls
            fixed_messages.append(new_msg)
        else:
            fixed_messages.append(msg)
    return fixed_messages

from langchain_core.messages import HumanMessage

# Ejemplo de ejecución del agente con una pregunta del usuario
messages = [HumanMessage(content="What is the price of gold in eur?")]
result = react_graph.invoke({"messages": messages})


# Mostrar el historial de mensajes resultante
for msg in result["messages"]:
    print(msg)

fixed_messages=fix_tool_calls_for_openai_format(result["messages"])
fixed_messages