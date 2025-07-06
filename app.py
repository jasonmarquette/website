import boto3
from botocore.config import Config
from flask import Flask, jsonify, request
import json

def init_bedrock_agent_runtime_client():
    return boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        config=Config(read_timeout=300),
    )

def init_bedrock_runtime_client():
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        config=Config(read_timeout=60),
    )

# Your Knowledge Base info
KNOWLEDGE_BASE_ID = "EVOLHELSIJ"
MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"

bedrock_client = init_bedrock_agent_runtime_client()
runtime_client = init_bedrock_runtime_client()

app = Flask(__name__)

FALLBACK_PHRASES = [
    "do not contain relevant data",
    "cannot find sufficient information",
    "no relevant data",
    "didn't find"
]

def query_knowledge_base(user_prompt):
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
    response = bedrock_client.retrieve_and_generate(**input_data)
    return response.get("output", {}).get("text", "")

def query_foundation_model(user_prompt):
    body = {
        "inputText": user_prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.7,
            "topP": 0.9,
        }
    }
    response = runtime_client.invoke_model(
        modelId="amazon.titan-text-express-v1",  # Use Titan for compatibility
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    result = json.loads(response['body'].read())
    return result.get("results", [{}])[0].get("outputText", "")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_prompt = data.get("prompt", "")
    if not user_prompt:
        return jsonify({"error": "Missing prompt"}), 400

    try:
        kb_response = query_knowledge_base(user_prompt)
        # If KB cannot answer, fallback to foundation model
        if not kb_response or any(phrase in kb_response.lower() for phrase in FALLBACK_PHRASES):
            fm_response = query_foundation_model(user_prompt)
            return jsonify({"response": fm_response, "source": "foundation_model"})
        else:
            return jsonify({"response": kb_response, "source": "knowledge_base"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)
