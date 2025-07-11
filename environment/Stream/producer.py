from kafka import KafkaProducer
import pandas as pd
import json
import time
import os
import boto3

# Kafka config
BROKER = 'localhost:9092'
TOPIC = 'covid-tweets'

# S3 config
BUCKET_NAME = 'x23411520-s3-bucket'
DATASET_KEY = 'covidvaccine.csv'
DATASET_PATH = '/home/ec2-user/environment/x23418613-c9-env/environment/Dataset/covidvaccine.csv'

# S3 Client
s3_client = boto3.client('s3')

# Download dataset from S3 only if not present locally
if not os.path.exists(DATASET_PATH):
    print(f"Dataset not found locally. Downloading from s3://{BUCKET_NAME}/{DATASET_KEY} ...")
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, DATASET_KEY, DATASET_PATH)
    print("ownload complete.")
else:
    print(f"Dataset already exists at {DATASET_PATH}, skipping download.")

# Load CSV
df = pd.read_csv(DATASET_PATH)
texts = df['text'].dropna().astype(str).tolist()

# Initialize producer
producer = KafkaProducer(
    bootstrap_servers=[BROKER],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

print(f"\n=====> Kafka Producer Started")
print(f"Streaming {len(texts)} tweets to topic '{TOPIC}'...\n")

# Stream tweets to Kafka
for text in texts:
    producer.send(TOPIC, value={"text": text})
    producer.flush()
    time.sleep(0.001)  # tiny delay to prevent backpressure on Cloud9

print("\n=====> All tweets sent successfully.")
