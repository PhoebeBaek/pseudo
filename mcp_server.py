from mcp.server.fastmcp import FastMCP
import config
import asyncio
import io
import base64
import vertexai
from vertexai.vision_models import Image, MultiModalEmbeddingModel
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_core.models import UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

# Initialize the MCP server
mcp = FastMCP("mcp_server")

@mcp.tool()
def input_image(image_path: str) -> str:
    """
    Receiving multimodal menu image and analyze the menu.
    
    Args:
        image_path: Path to the image file containing a menu
        
    Returns:
        Analysis about the menu.
    """
    # Open the uploaded image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Initialize the model client
    llm = ChatBedrock(
        region="us-east-1",
        provider = "anthropic",
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs = {"temperature": 1}
    )

    #Define a message
    message = HumanMessage(
        content=[
            {"type": "text", 
            "text": "You are a menu analyzer. Your task is to analyze the provided menu image and return a description about the menu."},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64_image
                }
            }
        ]
    )
    # Invoke LLM
    response = llm.invoke([message]).content
    return response

@mcp.tool()
def embed_image(image_path):
    # image_path = "/Users/sojeong/study/pseudo/image.png"
    embedding_dimension = 512
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    image = Image.load_from_file(
        image_path
    )

    embeddings = model.get_embeddings(
        image=image,
        dimension=embedding_dimension,
    )
    return embeddings.image_embedding

if __name__ == "__main__":
    mcp.run(transport='stdio')