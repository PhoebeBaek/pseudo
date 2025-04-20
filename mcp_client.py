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
        command="python", 
        args=["mcp_server.py"], 
        read_timeout_seconds=300
    )
    
    # Get the tools from the MCP server
    tools = await mcp_server_tools(server_params)

    # Initialize the model client
    model_client = OpenAIChatCompletionClient(
        model = "gemini-1.5-flash-8b",
        api_key = config.gemini_api_key
    )

    # Create the assistant agent with the tools
    agent = AssistantAgent(
        name = "assistant", 
        model_client = model_client, 
        tools = tools
    )
    ## image_path = "/Users/sojeong/study/pseudo/image.png"
    response = await agent.run(
        task=f"""
        Use the input_image tool with the image_path parameter set to '{image_path}'.
        Your task is analyzing ingredients required to cook the menu based on the give description from the tool.
                            
        """,
        cancellation_token=CancellationToken()
    )
    
    # Print the last message from the agent
    return response.messages[-1]

##asyncio.run(menu_analyze_agent(image_path = "/Users/sojeong/study/pseudo/image.png"))