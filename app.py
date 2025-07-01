import boto3
from botocore.config import Config
from flask import (
    Flask,
    jsonify,
    request,
)


def init_bedrock_client():
    """
    Initialize the Bedrock client with a longer read timeout.
    """
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        config=Config(read_timeout=300),
    )


bedrock_client = init_bedrock_client()


app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_prompt = data.get("prompt", "")

    # build the chat messages
    messages = [
        {
            "role": "user",
            "content": [{"text": user_prompt}],
        }
    ]

    # call the converse API
    response = bedrock_client.converse(
        modelId="amazon.nova-pro-v1:0",
        messages=messages,
        inferenceConfig={"temperature": 0.7, "topP": 0.9, "maxTokens": 1024},
    )

    # stitch together text chunks
    output_message = response["output"]["message"]
    text_parts = [chunk.get("text", "") for chunk in output_message["content"]]
    bot_response = "".join(text_parts)

    return jsonify({"response": bot_response})


if __name__ == "__main__":
    app.run(port=5000)
