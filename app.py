from flask import Flask, request, jsonify
import boto3
import json
from botocore.config import Config

def init_bedrock_client():
    """
    Initialize the Bedrock client with a longer read timeout.
    """
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",   # use your Bedrock region
        config=Config(read_timeout=300)
    )

# Create one global client
bedrock_client = init_bedrock_client()

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    user_prompt = data.get('prompt', '')

    # 1. Build the messages array for a chat model
    messages = [
        {
            "role": "user",
            "content": [
                {"text": user_prompt}
            ]
        }
    ]

    # 2. Call the Converse API instead of invoke_model
    response = bedrock_client.converse(
        modelId="amazon.nova-pro-v1:0",   # or whatever chat-capable model you prefer
        messages=messages,
        inferenceConfig={
            "temperature": 0.7,
            "topP": 0.9,
            "maxTokens": 1024
        }
    )

    # 3. Pull out the text chunks and stitch them together
    output_message = response["output"]["message"]
    text_parts = [chunk.get("text", "") for chunk in output_message["content"]]
    bot_response = "".join(text_parts)

    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(port=5000)
