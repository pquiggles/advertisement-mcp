import os
import asyncio
import anthropic
from typing import List, Dict, Optional
from dotenv import load_dotenv
import gradio as gr
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
import logging
import traceback

# Load environment variables
load_dotenv()

# Set up logging for the client
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('relationship-coach-client')

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# Enhanced system prompt
SYSTEM_PROMPT = """You are a compassionate and experienced relationship coach with expertise in:
- Communication strategies
- Conflict resolution
- Building healthy relationships
- Emotional intelligence
- Dating advice
- Marriage counseling
- Family dynamics

Your approach is:
- Empathetic and non-judgmental
- Solution-focused while validating emotions
- Evidence-based, drawing from psychology and relationship research
- Culturally sensitive and inclusive
- Protective of user privacy

You help users navigate relationship challenges by asking thoughtful questions,
providing practical advice, and offering tools for self-reflection and growth.

You have access to a curated database of products and services that some clients
find helpful. When someone mentions a specific need, challenge, or gift-giving occasion:
- Search for relevant products immediately rather than asking many questions first
- You can still ask clarifying questions AFTER showing initial suggestions
- Be proactive about finding helpful products when the context is clear

When suggesting products:
- Present 2-3 specific options with brief descriptions
- ALWAYS include the actual links so people can easily visit them
- Mention any available coupons
- Frame them as "here are some options that might work well"
- Keep the tone helpful and enthusiastic but not pushy

Always maintain appropriate boundaries and suggest professional therapy when issues
are beyond coaching scope.

IMPORTANT: When you receive product search results, they include 'url' fields with the actual links.
You MUST include these clickable links in your response so users can easily access the products.
Format them as: [Product Name](url) or simply share the full URL."""

# Conversation starters
EXAMPLE_PROMPTS = [
    "I want to find a thoughtful anniversary gift for my partner",
    "What are some good books about improving communication in relationships?",
    "I need ideas for date night activities that could help us reconnect",
    "Can you suggest tools or resources for managing relationship stress?",
]

async def test_mcp_connection():
    """Test the MCP connection independently"""
    try:
        logger.info("Testing MCP connection...")
        async with streamablehttp_client("http://localhost:8000/mcp") as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                logger.info(f"Connection successful! Available tools: {[tool.name for tool in tools.tools]}")
                return True
    except Exception as e:
        logger.error(f"Connection test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False

async def process_message_with_mcp(message: str, history: List[Dict]) -> str:
    """Process a message with MCP tools available to Claude"""
    
    # Build messages list
    messages = []
    for msg in history:
        if msg.get("role") in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})
    
    try:
        logger.info("Connecting to MCP server...")
        
        # Connect to the HTTP MCP server using streamable HTTP
        async with streamablehttp_client("http://localhost:8000/mcp") as (
            read_stream,
            write_stream,
            _,  # Third parameter we don't need
        ):
            logger.info("Transport streams created")
            
            # Create a session using the client streams
            async with ClientSession(read_stream, write_stream) as session:
                logger.info("Client session created")
                
                # Initialize the connection
                await session.initialize()
                logger.info("MCP session initialized successfully")
                
                # Get available tools from MCP server
                tools_response = await session.list_tools()
                available_tools = []
                
                for tool in tools_response.tools:
                    tool_def = {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema
                    }
                    available_tools.append(tool_def)
                    logger.info(f"Tool available: {tool.name}")
                
                logger.info(f"Processing message with {len(available_tools)} available tools")
                
                # Initial Claude API call with tools
                response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    temperature=0.7,
                    tools=available_tools if available_tools else None
                )
                
                # Process response and handle tool calls
                final_text = []
                assistant_message_content = []
                
                for content in response.content:
                    if content.type == 'text':
                        final_text.append(content.text)
                        assistant_message_content.append(content)
                    elif content.type == 'tool_use':
                        tool_name = content.name
                        tool_args = content.input
                        
                        logger.info(f"Calling tool: {tool_name} with args: {tool_args}")
                        
                        try:
                            # Execute tool call through MCP
                            result = await session.call_tool(tool_name, tool_args)
                            logger.info(f"Tool result type: {type(result)}")
                            logger.info(f"Tool result: {result}")
                            
                            # Add tool use to assistant message
                            assistant_message_content.append(content)
                            
                            # Add assistant message with tool use
                            messages.append({
                                "role": "assistant",
                                "content": assistant_message_content
                            })
                            
                            # Add tool result
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": str(result)
                                }]
                            })
                            
                            # Reset for next message
                            assistant_message_content = []
                            
                        except Exception as tool_error:
                            logger.error(f"Tool call error: {str(tool_error)}")
                            logger.error(traceback.format_exc())
                            
                            # Add error result
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": f"Tool error: {str(tool_error)}"
                                }]
                            })
                
                # Get final response if there were tool calls
                if any(content.type == 'tool_use' for content in response.content):
                    follow_up = anthropic_client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1000,
                        system=SYSTEM_PROMPT,
                        messages=messages,
                        temperature=0.7
                    )
                    
                    for content in follow_up.content:
                        if content.type == 'text':
                            final_text.append(content.text)
                
                return "\n".join(final_text) if final_text else "I couldn't process your request. Please try again."
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(traceback.format_exc())
        
        # Try to provide a helpful error message
        if "connection" in str(e).lower():
            return "I'm having trouble connecting to my product database. Please make sure the MCP server is running at http://localhost:8000/mcp"
        else:
            return f"I apologize, but I encountered an error: {str(e)}. Please try again or rephrase your request."

# Synchronous wrapper for Gradio
def chat_wrapper(message: str, history: List[Dict]) -> str:
    """Synchronous wrapper for the async chat function"""
    try:
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_message_with_mcp(message, history))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Chat wrapper error: {str(e)}")
        logger.error(traceback.format_exc())
        return f"An error occurred: {str(e)}"

# Create the Gradio interface
interface = gr.ChatInterface(
    fn=chat_wrapper,
    type="messages",
    title="💝 AI Relationship Coach (with Product Recommendations)",
    description="""Welcome! I'm here to help you navigate relationship challenges with empathy and evidence-based guidance.
    
    **What I can help with:** 
    - Communication issues and conflict resolution
    - Dating advice and relationship building
    - Gift ideas and product recommendations
    - Family dynamics and emotional support
    
    **Note:** While I provide coaching and support, I'm not a replacement for professional therapy.
    
    💡 **Try asking about:** Anniversary gifts, relationship books, date ideas, or communication tools!""",
    examples=EXAMPLE_PROMPTS,
)

if __name__ == "__main__":
    logger.info("Starting Relationship Coach Application")
    logger.info("Make sure the HTTP MCP server is running at http://localhost:8000/mcp")
    
    # Test the connection first
    test_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(test_loop)
    connection_ok = test_loop.run_until_complete(test_mcp_connection())
    test_loop.close()
    
    if not connection_ok:
        logger.warning("Initial connection test failed, but starting anyway...")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )