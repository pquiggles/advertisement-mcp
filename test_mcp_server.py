import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os

async def test_mcp_server():
    """Test the MCP affiliate server"""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "affiliate_mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()
            
            # List available tools
            response = await session.list_tools()
            tools = response.tools
            print("Available tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Test search_products
            print("\n\nTesting search_products for 'communication books':")
            result = await session.call_tool("search_products", {
                "query": "communication books for couples",
                "num_results": 3
            })
            print(result.content)
            
            # Test get_categories
            print("\n\nTesting get_categories:")
            result = await session.call_tool("get_categories", {})
            print(result.content)
            
            # Test get_top_products
            print("\n\nTesting get_top_products:")
            result = await session.call_tool("get_top_products", {
                "limit": 3
            })
            print(result.content)

if __name__ == "__main__":
    asyncio.run(test_mcp_server())