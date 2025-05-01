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
from pymongo import MongoClient

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
def mongodb_query(input_query: str):
    """
    Implement text to sql and run query in MongoDB.
    Args:
        input_query: natural language that needs to be converted into MongoDB query api.
        
    Returns:
        Query output return by MongoDB.
    """

    client = MongoClient(config.URI)
    db = client["dining-ai"]
    collection = db["items"]

    # Initialize the model client
    llm = ChatBedrock(
        region="us-east-1",
        # provider = "anthropic",
        # model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        model_id = "amazon.nova-pro-v1:0"
    )
    #Define a message
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": f"""
                    You are a MongoDB expert. Your task is converting the input query to MongoDB pymongo API following the rules.
                    Below is the schema sample in sample_mflix.movies namespace in MongoDB for your reference.
                    
                    <Input_query>
                    {input_query}
                    </Input_query>

                    <Rules>
                    1. Analzy the input thoroughly in order to understand what data needs to be found.
                    2. Return the query API method and filter only and do not add any explanation.
                    3. Do not add any extra filter on query.
                    4. Return the filter in valid JSON format.
                    5. Always exclude 'embedding' field in the query result.
                    6. Do not include any markdown formatting (like ```json) in your response.
                    </Rules>

                    <Example>
                    Input: Search for one movie which released in 2000 and with the rating higher than 4.
                    Return: find({{"year": 2000, "imdb.rating": {{"$gt": 4}}}}, {{"embedding": 0}}).limit(1)
                    </Example>

                    <Schema sample>
                    {{
                    "title": "The Arrival of a Train",
                    "year": {{
                        "$numberInt": "1896"
                    }},
                    "lastupdated": "2015-08-15 00:02:53.443000000",
                    "type": "movie",
                    "directors": [
                        "Auguste Lumière",
                        "Louis Lumière"
                    ],
                    "imdb": {{
                        "rating": {{
                        "$numberDouble": "7.3"
                        }},
                        "votes": {{
                        "$numberInt": "5043"
                        }},
                        "id": {{
                        "$numberInt": "12"
                        }}
                    }},
                    "cast": [
                        "Madeleine Koehler"
                    ],
                    "countries": [
                        "France"
                    ],
                    "genres": [
                        "Documentary",
                        "Short"
                    ],
                    "num_mflix_comments": {{
                        "$numberInt": "1"
                        }}
                    }}
                    </Schema sample>
                """
            }
        ]
    )

    # Invoke LLM
    mql = llm.invoke([message]).content
    response = eval(f"collection.{mql}")
    movie_list = []
    if response:
        for movie in response:
            movie_list.append(movie)
    return mql, movie_list


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