import boto3
from botocore.config import Config
from flask import Flask, jsonify, request


def init_bedrock_agent_runtime_client():
    """
    Initialize the Bedrock Agent Runtime client with a longer read timeout.
    """
    return boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        config=Config(read_timeout=300)
    )


# Set your KB ID and model ARN
KNOWLEDGE_BASE_ID = "EVOLHELSIJ"  # <-- Replace with your KB ID
MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"  # <-- Replace with your model ARN

bedrock_client = init_bedrock_agent_runtime_client()

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_prompt = data.get("prompt", "")

    input_data = {
        "input": {
            "text": user_prompt
        },
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": MODEL_ARN
            }
        }
    }

    try:
        response = bedrock_client.retrieve_and_generate(**input_data)
        bot_response = response.get("output", {}).get("text", "")
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if __name__ == "__main__":
    app.run(port=5000)
