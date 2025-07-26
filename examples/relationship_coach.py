import os
import gradio as gr
import anthropic
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

# Define the relationship coach system prompt
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
Always maintain appropriate boundaries and suggest professional therapy when issues 
are beyond coaching scope."""

# Conversation starters for users
EXAMPLE_PROMPTS = [
    "I'm having trouble communicating with my partner about household responsibilities",
    "How do I know if I'm ready to take the next step in my relationship?",
    "My family doesn't approve of my relationship. How should I handle this?",
    "I feel like I'm always the one making compromises. Is this normal?",
]


def chat_with_coach(message: str, history: List[Dict]) -> str:
    """
    Process user message and generate coach response using the Messages API
    
    Args:
        message: Current user message
        history: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Assistant's response
    """
    
    # Build messages list in the format expected by Claude
    messages = []
    
    # Add conversation history (already in correct format from Gradio)
    for msg in history:
        if msg.get("role") in ["user", "assistant"]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    try:
        # Make API call to Claude using the latest model
        response = client.messages.create(
            model="claude-opus-4-20250514",  # Latest Opus 4 model
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages,
            temperature=0.7,  # Balanced creativity and consistency
        )
        
        return response.content[0].text
        
    except anthropic.APIError as e:
        return f"I apologize, but I'm having trouble connecting. API Error: {str(e)}"
    except Exception as e:
        return f"I apologize, but something went wrong. Error: {str(e)}"


# Create the simplest possible interface
interface = gr.ChatInterface(
    fn=chat_with_coach,
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