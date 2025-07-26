import os
import asyncio
import anthropic
from typing import List, Dict, Optional
from dotenv import load_dotenv
import gradio as gr
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load environment variables
load_dotenv()

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
    "I'm having trouble communicating with my partner about household responsibilities",
    "How do I know if I'm ready to take the next step in my relationship?",
    "My family doesn't approve of my relationship. How should I handle this?",
    "I feel like I'm always the one making compromises. Is this normal?",
]

async def process_message_with_mcp(message: str, history: List[Dict]) -> str:
    """Process a message with MCP tools available to Claude"""
    
    # Build messages list
    messages = []
    for msg in history:
        if msg.get("role") in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})
    
    # Connect to MCP server
    server_params = StdioServerParameters(
        command="uv",
        args=[
            "--directory",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "run",
            "affiliate_mcp_server.py"
        ]
    )
    
    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:
            await session.initialize()
            
            # Get available tools from MCP server
            tools_response = await session.list_tools()
            available_tools = [{
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            } for tool in tools_response.tools]
            
            try:
                # Initial Claude API call with tools
                response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    temperature=0.7,
                    tools=available_tools
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
                        
                        # Execute tool call through MCP
                        result = await session.call_tool(tool_name, tool_args)
                        
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
                                "content": str(result.content)
                            }]
                        })
                        
                        # Get Claude's response after tool use
                        follow_up = anthropic_client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=1000,
                            system=SYSTEM_PROMPT,
                            messages=messages,
                            temperature=0.7
                        )
                        
                        final_text.append(follow_up.content[0].text)
                
                return "\n".join(final_text)
                
            except anthropic.APIError as e:
                return f"I apologize, but I'm having trouble connecting. API Error: {str(e)}"
            except Exception as e:
                return f"I apologize, but something went wrong. Error: {str(e)}"

# Synchronous wrapper for Gradio
def chat_wrapper(message: str, history: List[Dict]) -> str:
    """Synchronous wrapper for the async chat function"""
    return asyncio.run(process_message_with_mcp(message, history))

# Create the Gradio interface
interface = gr.ChatInterface(
    fn=chat_wrapper,
    type="messages",
    title="💝 AI Relationship Coach",
    description="""Welcome! I'm here to help you navigate relationship challenges with empathy and evidence-based guidance.
    
    **What I can help with:** Communication issues, conflict resolution, dating advice, family dynamics, and building healthier relationships.
    
    **Note:** While I provide coaching and support, I'm not a replacement for professional therapy.""",
    examples=EXAMPLE_PROMPTS,
)

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )