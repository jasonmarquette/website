import json
import logging
import threading

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from flask import Flask, jsonify, request


# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# Bedrock Clients (NO RETRIES)
# -------------------------------------------------
def init_bedrock_agent_runtime_client():
    return boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        config=Config(
            connect_timeout=5,
            read_timeout=30,
            retries={"max_attempts": 0},
        ),
    )


def init_bedrock_runtime_client():
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        config=Config(
            connect_timeout=5,
            read_timeout=30,
            retries={"max_attempts": 0},
        ),
    )


bedrock_agent_client = init_bedrock_agent_runtime_client()
bedrock_runtime_client = init_bedrock_runtime_client()


# -------------------------------------------------
# App config
# -------------------------------------------------
app = Flask(__name__)

KNOWLEDGE_BASE_ID = "EVOLHELSIJ"

# REQUIRED for RetrieveAndGenerate
KB_MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-pro-v1:0"

# VALID runtime fallback model
FALLBACK_MODEL_ID = "amazon.nova-lite-v1:0"

KB_HARD_TIMEOUT = 6  # seconds


# -------------------------------------------------
# Knowledge Base (thread + hard timeout)
# -------------------------------------------------
def _kb_worker(prompt: str, result: dict):
    try:
        logger.info("KB: calling retrieve_and_generate")

        response = bedrock_agent_client.retrieve_and_generate(
            input={"text": prompt},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": KB_MODEL_ARN,
                },
            },
        )

        result["text"] = response.get("output", {}).get("text", "")

    except ClientError as e:
        if e.response["Error"]["Code"] == "ThrottlingException":
            logger.warning("KB: throttled, skipping")
        else:
            logger.exception("KB: client error")
        result["text"] = ""

    except Exception:
        logger.exception("KB: unexpected error")
        result["text"] = ""


def query_knowledge_base(prompt: str) -> str:
    result = {"text": ""}
    t = threading.Thread(target=_kb_worker, args=(prompt, result), daemon=True)
    t.start()
    t.join(timeout=KB_HARD_TIMEOUT)

    if t.is_alive():
        logger.error("KB: hard timeout reached")
        return ""

    return result["text"]


# -------------------------------------------------
# Foundation Model fallback (VALID MODEL)
# -------------------------------------------------
def query_foundation_model(prompt: str) -> str:
    try:
        logger.info("FM: invoking fallback model")

        response = bedrock_runtime_client.invoke_model(
            modelId=FALLBACK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 512,
                        "temperature": 0.7,
                        "topP": 0.9,
                    },
                }
            ),
        )

        payload = json.loads(response["body"].read())
        return payload.get("results", [{}])[0].get("outputText", "")

    except Exception:
        logger.exception("FM: failed")
        return "Sorry — the assistant is temporarily unavailable."


# -------------------------------------------------
# Route
# -------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    logger.info("CHAT: request received")

    kb_text = query_knowledge_base(prompt)

    if not kb_text:
        logger.info("CHAT: using fallback model")
        fm_text = query_foundation_model(prompt)
        return jsonify({"response": fm_text, "source": "foundation_model"})

    return jsonify({"response": kb_text, "source": "knowledge_base"})


# -------------------------------------------------
# Local dev only
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
