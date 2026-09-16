import os
import boto3

BUCKET = "stock-news-models-654639800551-654639800551-us-east-1-an"
S3_PREFIX = "models/"
MODEL_DIR = "/app/models"

s3 = boto3.client("s3")

os.makedirs(MODEL_DIR, exist_ok=True)

for filename in ["mistral-7b-instruct-v0.2.Q4_K_M.gguf", "mistral-7b-instruct-v0.2.Q8_0.gguf"]:
    local_path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(local_path):
        print(f"{filename} already exists, skippig")
        continue

    print(f"Downloading {filename}")

    s3.download_file(
        BUCKET,
        S3_PREFIX + filename,
        local_path
    )

    print(f"Downloaded {filename}")