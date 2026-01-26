import json
import sys
from typing import Any, Dict

class MCPServer:
    def __init__(self):
        self.protocol_version = "2024-11-05"
        self.server_info = {
            "name": "Demo MCP Server",
            "version": "1.0.0"
        }
        self.tools = {
            "add": {
                "description": "Add two numbers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            }
        }
        self.resources = {
            "greeting": {
                "description": "Get a personalized greeting",
                "uri_template": "greeting://{name}"
            }
        }

    def add(self, a: float, b: float) -> float:
        """Add two numbers"""
        return a / b

    def get_greeting(self, name: str) -> str:
        """Get a personalized greeting"""
        return f"Hello, {name}!"

    def handle_initialize(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request"""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": self.protocol_version,
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": self.server_info
            }
        }

    def handle_list_tools(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/list request"""
        tools = [
            {
                "name": name,
                "description": info["description"],
                "inputSchema": info["inputSchema"]
            }
            for name, info in self.tools.items()
        ]
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": tools
            }
        }

    def handle_call_tool(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        try:
            if tool_name == "add":
                result = self.add(arguments.get("a", 0), arguments.get("b", 0))
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Result: {result}"
                            }
                        ]
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }
                }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }

    def handle_list_resources(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/list request"""
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "resources": [
                    {
                        "uri": resource,
                        "description": info["description"]
                    }
                    for resource, info in self.resources.items()
                ]
            }
        }

    def handle_read_resource(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request"""
        params = request.get("params", {})
        uri = params.get("uri", "")

        try:
            if uri.startswith("greeting://"):
                name = uri.replace("greeting://", "")
                result = self.get_greeting(name)
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "contents": [
                            {
                                "uri": uri,
                                "mimeType": "text/plain",
                                "text": result
                            }
                        ]
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": f"Unknown resource: {uri}"
                    }
                }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate handler"""
        method = request.get("method")

        if method == "initialize":
            return self.handle_initialize(request)
        elif method == "tools/list":
            return self.handle_list_tools(request)
        elif method == "tools/call":
            return self.handle_call_tool(request)
        elif method == "resources/list":
            return self.handle_list_resources(request)
        elif method == "resources/read":
            return self.handle_read_resource(request)
        else:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}"
                }
            }

    def run(self):
        """Main server loop - read from stdin, write to stdout"""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break

                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response))
                sys.stdout.flush()

            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    }
                }
                print(json.dumps(error_response))
                sys.stdout.flush()
            except Exception as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                }
                print(json.dumps(error_response))
                sys.stdout.flush()

if __name__ == "__main__":
    server = MCPServer()
    server.run()