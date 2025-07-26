#!/usr/bin/env python3
"""
Quick test script to verify the relationship coach MCP connection works
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_connection():
    """Test the MCP connection"""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    
    print("Testing MCP connection to http://localhost:8000/mcp...")
    
    try:
        async with streamablehttp_client("http://localhost:8000/mcp") as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                
                # List available tools
                tools = await session.list_tools()
                print(f"✓ Connected! Found {len(tools.tools)} tools:")
                for tool in tools.tools:
                    print(f"  - {tool.name}")
                
                # Test a search
                print("\nTesting product search...")
                result = await session.call_tool("search_products", {
                    "query": "anniversary gift",
                    "num_results": 3
                })
                
                print(f"✓ Search successful! Result type: {type(result)}")
                
                # Check if result has content
                if hasattr(result, 'content'):
                    print(f"  Content type: {type(result.content)}")
                    if isinstance(result.content, list):
                        print(f"  Number of items: {len(result.content)}")
                
                return True
                
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # First check if MCP server is running
    import requests
    try:
        response = requests.get('http://localhost:8000/mcp', timeout=2)
        print("✓ MCP server is running")
    except:
        print("✗ MCP server is not running!")
        print("Start it with: python affiliate_mcp_server.py")
        exit(1)
    
    # Test the connection
    success = asyncio.run(test_connection())
    
    if success:
        print("\n✓ All tests passed! The relationship coach should work now.")
    else:
        print("\n✗ Tests failed. Check the errors above.")