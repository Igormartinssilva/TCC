from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")

# Wrap FastMCP com FastAPI
app = FastAPI()

# CORS para VS Code acessar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Melhorar descrições para detecção automática
@mcp.tool()
def add(a: int, b: int) -> int:
    """
    Add two numbers together.
    
    Use this tool when:
    - User asks to add, sum, or calculate the total of numbers
    - User needs to perform addition operations
    - User mentions combining or totaling numeric values
    
    Args:
        a: First integer to add
        b: Second integer to add
    
    Returns:
        The sum of a and b
    """
    return a + b  # Corrigido: estava retornando a - b

@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """
    Get a personalized greeting message.
    
    Use this resource when:
    - User asks for a greeting or welcome message
    - User wants to greet someone by name
    - User needs a personalized message
    
    Args:
        name: The name of the person to greet
    """
    return f"Hello, {name}!"

@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
    """
    Generate a greeting prompt template.
    
    Use this prompt when:
    - User wants to create a greeting message
    - User needs help writing a greeting
    - User asks for greeting templates or examples
    
    Args:
        name: Name of the person to greet
        style: Style of greeting (friendly, formal, casual)
    """
    styles = {
        "friendly": "Please write a warm, friendly greeting",
        "formal": "Please write a formal, professional greeting",
        "casual": "Please write a casual, relaxed greeting",
    }
    return f"{styles.get(style, styles['friendly'])} for someone named {name}."
# Rotas MCP expostas como HTTP
@app.get("/tools")
def list_tools():
    return {"tools": [{"name": "add", "description": "Add two numbers"}]}

@app.post("/tools/add")
def call_add(a: int, b: int):
    return {"result": add(a, b)}

@app.get("/resources/greeting/{name}")
def call_greeting(name: str):
    return {"greeting": get_greeting(name)}

if __name__ == "__main__":
    # Rodar servidor na porta 8090
    uvicorn.run(app, host="127.0.0.1", port=8090)