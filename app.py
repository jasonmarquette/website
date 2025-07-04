import boto3
from botocore.config import Config
from flask import Flask, jsonify, request
import logging

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console
        logging.FileHandler("bedrock_flask_debug.log")  # File
    ]
)

print("=== Flask Bedrock KB Debug Server Starting ===")
logging.info("=== Flask Bedrock KB Debug Server Starting ===")

def init_bedrock_agent_runtime_client():
    return boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        config=Config(read_timeout=300)
    )

# Replace with your actual Knowledge Base ID and Model ARN
KNOWLEDGE_BASE_ID = "EVOLHELSIJ"
MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"

bedrock_client = init_bedrock_agent_runtime_client()

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    print("DEBUG: /CHAT endpoint called")
    logging.info("DEBUG: /CHAT endpoint called")
    data = request.get_json() or {}
    user_prompt = data.get("prompt", "")

    input_data = {
        "input": {"text": user_prompt},
        "retrieveAndGenerateConfiguration": {
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                "modelArn": MODEL_ARN,
            },
        },
    }

    try:
        response = bedrock_client.retrieve_and_generate(**input_data)
        logging.info("\n===== FULL BEDROCK RESPONSE =====\n%s\n==============================\n", response)
        print("\n===== FULL BEDROCK RESPONSE =====")
        print(response)
        print("=================================\n")
        bot_response = response.get("output", {}).get("text", "")
        return jsonify({"response": bot_response})
    except Exception as e:
        logging.error("ERROR: %s", str(e))
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000,debug=True)
