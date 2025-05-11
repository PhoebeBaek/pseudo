import os
from mcp.server.fastmcp import FastMCP
import config
import asyncio
import io
import re
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
def input_image(image_path):
    """
    Receiving multimodal menu image and analyze the menu.
    
    Args:
        image_path: Path to the image file containing a menu.
        
    Returns:
        Ingredients analysis about the menu.
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
            "text": """
                    You are a menu analyzer. Your are responsible for the tasks based on the rules below.
                    <Tasks>
                    1. Analyze menu images
                    2. Analyze what ingredients are required to cook the menu in Korean.
                    </Tasks>

                    <Rules>
                    1. Return ONLY a Python list of ingredients, nothing else
                    2. Each element in array should be a string
                    3. Do not include any other text or formatting
                    </Rules>
                    """
            },
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
def mongodb_query(input_items):
    """
    Implement text to sql and run query in MongoDB.
    Args:
        input_items: String representation of a list of items from `menu_analysis_assistant`.
        
    Returns:
        Query output return by MongoDB.
    """

    client = MongoClient(config.URI)
    db = client["dining_ai"]
    collection = db["items"]

    # Initialize the model client
    llm = ChatBedrock(
        region="us-east-1",
        provider = "anthropic",
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        # model_id = "amazon.nova-pro-v1:0"
    )
    #Define a message
    message = HumanMessage(
        content=[
            {
                "type": "text",
                    "text": f"""
                    You are a MongoDB expert. Your task is writing MongoDB aggregation API to query a item based on the input_items.
                    The input_items is {input_items}.
                    Below is the schema sample in dining_ai.items namespace in MongoDB for your information.

                    <Rules>
                        1. Find the first item in input_items to understand what data needs to be found. If the input_items is ['사과', '부추'], only '사과' is used to query the item.
                        2. Create a MongoDB aggregation API for only the first item based on the input.
                        3. Add limit stage to limit the number of results to 3.
                        4. Return MongoDB aggregation API in valid JSON format.
                        5. Do not include any markdown formatting and additional filters.
                    </Rules>
                    <Example>
                    Input: {{
                                "source":"menu_analysis_assistant"
                                "models_usage":NULL
                                "metadata":{{}}
                                "content":"[TextContent(type='text', text="['오징어', '부추']", annotations=None)]"
                                "type":"ToolCallSummaryMessage"
                            }}
                    Return: [{{'$search':{{'text':{{'query':'고기','path':'title'}}}}}},{{'$limit':3}}]
                    </Example>

                    <Schema sample>
                    {{"_id":{{"$oid":"681715a9cc50fd56598796f6"}},"title":"미국산 프라임 척아이롤(목심+등심) 100G/소고기","lprice":"16380","hprice":"","mallName":"Homeplus","productId":"82539599247","productType":"2","brand":"","maker":"","category1":"식품","category2":"축산물","category3":"쇠고기","category4":"수입산쇠고기"}}
                    </Schema sample>
                """
            }
        ]
    )

    # Invoke LLM
    mql = llm.invoke([message]).content
    print(mql)
    response = eval(f"collection.aggregate({mql})")
    item_list = []
    for item in response:
        item_list.append(item)
    print(item_list)
    return item_list


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