from flask import Flask, request, jsonify
import boto3
import json
import boto3

# To run this code, ensure you have Flask and boto3 installed:
# pip install Flask boto3
# Also, ensure you have configured your AWS credentials properly.
# You can set them up using the AWS CLI or by creating a credentials file at ~/.aws
# /credentials.
# The credentials file should look like this:
# [default]
# aws_access_key_id = YOUR_ACCESS_KEY
# aws_secret_access_key = YOUR_SECRET

# In my case, I put the credentials in the service.


bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"   # use your Bedrock region
)


app = Flask(__name__)
boto3.client("bedrock-runtime", region_name="us-east-1")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_prompt = data.get('prompt', '')

    response = bedrock_client.invoke_model(
        modelId='amazon.titan-text-express-v1',
        contentType='application/json',
        accept='application/json',
        body=json.dumps({
            "inputText": user_prompt,
            "textGenerationConfig": {
                "maxTokenCount": 3072,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9,
            }
        })
    )

    raw_body = response['body'].read()
    response_data = json.loads(raw_body)
    output_text = response_data.get("results", [])[0].get("outputText", "No response received.")

    return jsonify({"response": output_text})


if __name__ == "__main__":
    app.run(port=5000)
