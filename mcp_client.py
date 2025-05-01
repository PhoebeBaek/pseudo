import config
import asyncio
from pathlib import Path
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_core import CancellationToken


async def menu_analyze_agent(image_path) -> str:
    ## file_directory = str(Path.home() / "study" / "pseudo")
    server_params = StdioServerParams(
        command = "python", 
        args = ["mcp_server.py"], 
        read_timeout_seconds = 300
    )
    
    # Get the tools from the MCP server
    tools = await mcp_server_tools(server_params)

    # Initialize the model client
    model_client = OpenAIChatCompletionClient(
        model = "gemini-1.5-pro",
        api_key = config.gemini_api_key
    )

    # Create the assistant agent with the tools
    agent = AssistantAgent(
        name = "assistant", 
        model_client = model_client, 
        tools = tools,
        system_message = """You are a menu analysis expert. Your task is to:
        1. Analyze menu images to identify ingredients
        2. Extract ONLY the ingredients mentioned in the description
        3. Return them as a Python list of strings
        4. Do not include any other text or formatting
        5. You MUST provide a final response after using any tools"""
    )
    ## image_path = "/Users/sojeong/study/pseudo/image.png"
    response = await agent.run(
        task = f"""
        Follow these steps:
        1. Use the input_image tool with the image_path parameter set to '{image_path}' to get the menu description
        2. Analyze the description and extract ONLY the ingredients mentioned
        3. Return the ingredients as a Python list of strings

        Rules:
        1. Return ONLY a Python list of ingredients, nothing else
        2. Each ingredient should be a string
        3. Do not include any other text or formatting
        4. After getting the description from input_image, you MUST process it into a list of ingredients
        5. You MUST provide a final response with the list of ingredients
                            
        Example valid responses:
        ["chicken", "rice", "soy sauce"]
        ["beef", "onion", "garlic", "salt"]                    
        """,
        cancellation_token = CancellationToken()
    )
    
    # Print all messages to debug
    print("\nAll messages from the agent:")
    for msg in response.messages:
        print(f"\nMessage type: {type(msg)}")
        print(f"Message content: {msg.content}")
    
    return response.messages
    

##asyncio.run(menu_analyze_agent(image_path = "/Users/sojeong/study/pseudo/image.png"))