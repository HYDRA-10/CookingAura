import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

async def setup_mcp_tools():
    """Inicia el servidor MCP como subproceso y carga sus herramientas."""
    server_params = StdioServerParameters(
        command="python",
        args=["src/agents/mcp_tool.py"], 
    )
    
    exit_stack = AsyncExitStack()
    read, write = await exit_stack.enter_async_context(stdio_client(server_params))
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    
    await session.initialize()
    tools = await load_mcp_tools(session)
    
    return tools, exit_stack

async def test_mcp_connection():
    print("Iniciando servidor MCP y conectando...")
    tools, exit_stack = await setup_mcp_tools()
    
    try:
        # Extraemos la herramienta que LangChain ha generado automáticamente
        ingredientes_tool = tools[0]
        print(f"✅ Herramienta cargada con éxito: {ingredientes_tool.name}\n")
        
        # Prueba 1: Simular que el LLM pide todos los ingredientes
        print("--- PRUEBA 1: Consultar toda la despensa ---")
        respuesta_total = await ingredientes_tool.ainvoke({})
        print(respuesta_total)
        print("-" * 40 + "\n")
        
        # Prueba 2: Simular que el LLM filtra por 'verdura'
        print("--- PRUEBA 2: Consultar solo verduras ---")
        respuesta_verduras = await ingredientes_tool.ainvoke({"category": "verdura"})
        print(respuesta_verduras)
        print("-" * 40 + "\n")
        
    finally:
        # Es vital cerrar el AsyncExitStack para matar el subproceso del servidor
        await exit_stack.aclose()
        print("Conexión MCP cerrada correctamente.")

if __name__ == "__main__":
    asyncio.run(test_mcp_connection())