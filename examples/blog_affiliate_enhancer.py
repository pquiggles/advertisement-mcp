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
import re

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('blog-affiliate-client')

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# System prompt for subtle affiliate link insertion
SYSTEM_PROMPT = """You are an expert content editor specializing in natural affiliate link integration. Your task is to enhance blog posts by subtly incorporating relevant product recommendations with affiliate links.

Your approach:
- Make minimal changes to preserve the author's voice and style
- Only add product mentions where they genuinely add value
- Keep insertions natural and contextually appropriate
- Maintain the original flow and readability
- Never be overly promotional or sales-focused
- Aim for 2-4 affiliate link insertions per 500 words

You have access to a product search tool. When enhancing a blog post:
1. Search for relevant products based on the blog's topics
2. Select 2-4 products from the search results that genuinely fit the content
3. Integrate these products naturally using the EXACT URLs from the 'url' field
4. Include coupon information if available in the 'coupon' field

CRITICAL RULES:
- You MUST use the search_products tool to find products
- You MUST ONLY use products that appear in the search results
- You MUST use the EXACT URLs provided in the 'url' field
- NEVER create or modify URLs
- NEVER use placeholder URLs or generate Amazon links

When you receive search results and need to provide the final enhanced blog post, output ONLY the blog post content with affiliate links integrated. Do not include any explanations or meta-commentary."""

# Example blog posts
EXAMPLE_POSTS = [
    """# 5 Tips for Better Sleep

Getting quality sleep is essential for our health and wellbeing. Here are my top tips for improving your sleep quality:

1. **Stick to a Schedule**: Try to go to bed and wake up at the same time every day, even on weekends.

2. **Create a Relaxing Environment**: Keep your bedroom cool, dark, and quiet. 

3. **Limit Screen Time**: Avoid phones and computers for at least an hour before bed.

4. **Watch Your Diet**: Avoid large meals, caffeine, and alcohol before bedtime.

5. **Stay Active**: Regular physical activity can help you fall asleep faster and enjoy deeper sleep.

Remember, good sleep is not a luxury—it's a necessity for optimal health!""",
    
    """# Starting Your Home Garden

Spring is the perfect time to start a home garden. Whether you have a large backyard or just a small balcony, you can grow your own fresh produce.

First, assess your space and sunlight. Most vegetables need at least 6 hours of direct sunlight. If you're limited on space, consider container gardening.

Choose easy-to-grow plants for beginners like tomatoes, lettuce, and herbs. These are forgiving and provide quick results to keep you motivated.

Prepare your soil properly. Good soil is the foundation of a healthy garden. Mix in compost to provide nutrients for your plants.

Water consistently but don't overdo it. Most plants prefer deep, infrequent watering rather than daily sprinkles.

Happy gardening!""",
    
    """# Minimalist Living: A Beginner's Guide

Minimalism isn't about having nothing—it's about having just enough. Here's how to start your minimalist journey:

**Start Small**: Begin with one drawer or shelf. Sort items into keep, donate, and discard piles.

**One In, One Out Rule**: When you bring something new into your home, remove something else.

**Digital Declutter**: Don't forget about digital clutter. Organize your files and unsubscribe from unnecessary emails.

**Quality Over Quantity**: Invest in fewer, better-quality items that will last longer.

**Mindful Purchases**: Before buying, ask yourself if you really need it and where it will live in your home.

The goal is to surround yourself only with things that support your current life and bring you joy."""
]

async def process_blog_with_mcp(blog_content: str) -> str:
    """Process blog post and add affiliate links using MCP tools"""
    
    try:
        logger.info("Connecting to MCP server for blog processing...")
        
        # Connect to the HTTP MCP server
        async with streamablehttp_client("http://localhost:8000/mcp") as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                logger.info("MCP session initialized")
                
                # Get available tools
                tools_response = await session.list_tools()
                available_tools = []
                
                for tool in tools_response.tools:
                    tool_def = {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.inputSchema
                    }
                    available_tools.append(tool_def)
                
                # Initial message to Claude
                messages = [{
                    "role": "user",
                    "content": f"""Please enhance this blog post by subtly adding 2-4 relevant affiliate product recommendations. Search for products that would genuinely help readers and integrate them naturally into the content.

Blog post to enhance:

{blog_content}"""
                }]
                
                # First Claude API call
                response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    temperature=0.7,
                    tools=available_tools if available_tools else None
                )
                
                # Process response and handle tool calls
                assistant_message_content = []
                has_tool_use = False
                
                for content in response.content:
                    if content.type == 'text':
                        assistant_message_content.append(content)
                    elif content.type == 'tool_use':
                        has_tool_use = True
                        tool_name = content.name
                        tool_args = content.input
                        
                        logger.info(f"Searching for products: {tool_args}")
                        
                        try:
                            # Execute tool call
                            result = await session.call_tool(tool_name, tool_args)
                            
                            # Log the tool response for debugging
                            logger.info(f"Tool response received: {result.content}")
                            
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
                            
                            # Reset assistant_message_content for next iteration
                            assistant_message_content = []
                            
                        except Exception as tool_error:
                            logger.error(f"Tool error: {str(tool_error)}")
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": f"Tool error: {str(tool_error)}"
                                }]
                            })
                
                # Get final response if there were tool calls
                if has_tool_use:
                    # Add a reminder in the final request
                    messages.append({
                        "role": "user",
                        "content": "Now please provide the enhanced blog post with the affiliate links naturally integrated. Output only the blog post content, no explanations."
                    })
                    
                    follow_up = anthropic_client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=2000,
                        system=SYSTEM_PROMPT,
                        messages=messages,
                        temperature=0.7
                    )
                    
                    # Extract the final enhanced blog post
                    enhanced_content = ""
                    for content in follow_up.content:
                        if content.type == 'text':
                            enhanced_content = content.text
                            break
                else:
                    # No tool use, just return the text response
                    enhanced_content = ""
                    for content in response.content:
                        if content.type == 'text':
                            enhanced_content = content.text
                            break
                
                if not enhanced_content:
                    enhanced_content = blog_content
                
                # Add a subtle indicator of affiliate links
                affiliate_notice = "\n\n---\n*This post contains affiliate links. We may earn a commission if you make a purchase through these links, at no extra cost to you.*"
                
                return enhanced_content + affiliate_notice
        
    except Exception as e:
        logger.error(f"Error processing blog: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error processing blog post: {str(e)}\n\nOriginal content:\n{blog_content}"

def process_blog_wrapper(blog_content: str) -> str:
    """Synchronous wrapper for Gradio"""
    try:
        if not blog_content.strip():
            return "Please enter a blog post to enhance with affiliate links."
        
        # Show processing message
        logger.info(f"Processing blog post of {len(blog_content)} characters")
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_blog_with_mcp(blog_content))
        loop.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Wrapper error: {str(e)}")
        return f"An error occurred: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="Blog Affiliate Link Enhancer") as interface:
    gr.Markdown("""
    # 📝 Blog Post Affiliate Link Enhancer
    
    Transform your blog posts with subtle, relevant affiliate product recommendations. 
    Our AI will naturally integrate 2-4 product mentions that add genuine value to your content.
    
    **How it works:**
    1. Paste your blog post below
    2. Our AI analyzes the content and finds relevant product opportunities
    3. Get an enhanced version with naturally integrated affiliate links
    
    **Features:**
    - Preserves your writing style and voice
    - Only adds products where they genuinely help readers
    - Includes real affiliate links with available coupons
    - Maintains natural flow and readability
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Original Blog Post",
                placeholder="Paste your blog post here...",
                lines=15,
                max_lines=30
            )
            
            gr.Examples(
                examples=EXAMPLE_POSTS,
                inputs=input_text,
                label="Try an example post:"
            )
            
            process_btn = gr.Button("✨ Enhance with Affiliate Links", variant="primary", size="lg")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Enhanced Blog Post",
                lines=15,
                max_lines=30,
                show_copy_button=True
            )
    
    # Stats display
    with gr.Row():
        gr.Markdown("""
        ### 📊 What to expect:
        - **2-4 product mentions** per 500 words
        - **Natural integration** that doesn't disrupt flow
        - **Relevant products** that truly benefit readers
        - **Higher engagement** and potential revenue
        """)
    
    process_btn.click(
        fn=process_blog_wrapper,
        inputs=input_text,
        outputs=output_text
    )
    
    gr.Markdown("""
    ---
    **Note:** Make sure the MCP affiliate server is running at `http://localhost:8000/mcp`
    
    **Tips for best results:**
    - Write naturally about topics that lend themselves to product recommendations
    - Longer posts (300+ words) work better for natural integration
    - Posts about hobbies, lifestyle, how-tos, and reviews work particularly well
    """)

if __name__ == "__main__":
    logger.info("Starting Blog Affiliate Enhancer")
    logger.info("Make sure the MCP server is running at http://localhost:8000/mcp")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from relationship coach
        share=False,
        show_error=True,
    )