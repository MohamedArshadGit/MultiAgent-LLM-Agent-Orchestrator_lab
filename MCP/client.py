from langchain_mcp_adapters.client import MultiServerMCPClient
#from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client =MultiServerMCPClient(
        {
            "Math":{"command":"python",
            "args":[r"C:\Users\Mohamed Arshad\Downloads\My_RAG_Lab\MultiAgent-LLM-Agent-Orchestrator_lab\MCP\mathserver.py"],
            "transport":"stdio"},

            "Weather":{
            "url":"http://127.0.0.1:8000/mcp",
            "transport":"streamable-http"
            }
        }
    )

    import os
    os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

    tools=await client.get_tools()
    model =ChatGroq(model="openai/gpt-oss-120b")
    # agent =create_react_agent(model,tools)
    agent =create_agent(model,tools)


    math_response =await agent.ainvoke(
        {"messages":[{"role":"user","content":"whats (3+7) *20"}]})
    print("math_response:",math_response['messages'][-1].content)

    weather_response =await agent.ainvoke(
        {"messages":[{"role":"user","content":"whats weather in antartica"}]})
    print("weather_response:",weather_response['messages'][-1].content)

asyncio.run(main())