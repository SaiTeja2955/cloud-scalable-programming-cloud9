import json
from kafka import KafkaConsumer
from collections import Counter, deque
from multiprocessing import Pool, cpu_count
from textblob import TextBlob
import psutil
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import boto3
import pandas as pd

# ==== CONFIG ====
TOPIC = 'covid-tweets'
BROKER = 'localhost:9092'

WINDOW_SIZE = 5000
WINDOW_SLIDE = 2500

TOP_N = 10
RESULT_DIR = '/home/ec2-user/environment/x23418613-c9-env/environment/results_2'
DATASET_PATH = '/home/ec2-user/environment/x23418613-c9-env/environment/Dataset/covidvaccine.csv'
BUCKET_NAME = 'x23411520-s3-bucket'
DATASET_KEY = 'covidvaccine.csv'


WORKLOADS = [10000, 100000, 200000]

# ==== SETUP ====
os.makedirs(RESULT_DIR, exist_ok=True)

# S3 Client
s3_client = boto3.client('s3')

# Download dataset from S3 if not present locally
if not os.path.exists(DATASET_PATH):
    print(f"Dataset not found locally. Downloading from s3://{BUCKET_NAME}/{DATASET_KEY} ...")
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, DATASET_KEY, DATASET_PATH)
    print("Download complete.")
else:
    print(f"Dataset already exists at {DATASET_PATH}, skipping download.")

window = deque(maxlen=WINDOW_SIZE)

def clean_and_tokenize(text):
    import re
    return re.findall(r'\b[a-z]+\b', text.lower())

def extract_hashtags_mentions(text):
    words = text.split()
    return [w for w in words if w.startswith('#') or w.startswith('@')]

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def process_tweet(text):
    words = clean_and_tokenize(text)
    tags = extract_hashtags_mentions(text)
    sentiment = get_sentiment(text)
    word_count = len(words)
    return words, tags, len(text), sentiment, word_count

def parallel_process(data):
    with Pool(processes=cpu_count()) as pool:
        return pool.map(process_tweet, data)

def safe_deserialize(x):
    try:
        return json.loads(x.decode('utf-8')) if x else {}
    except:
        return {}

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=BROKER,
    auto_offset_reset='earliest',
    enable_auto_commit=False,
    value_deserializer=safe_deserialize
)

print("\n=====> Kafka Stream Processing Started (Sequential + Parallel modes for each workload)\n")

batch = []
current_workload_index = 0
next_workload_target = WORKLOADS[current_workload_index]

results_per_workload = []

hashtag_counter_seq = Counter()
hashtag_counter_par = Counter()

for msg in consumer:
    if 'text' not in msg.value:
        continue

    tweet_text = msg.value['text']
    batch.append(tweet_text)
    window.append(tweet_text)

    words, tags, length, sentiment, word_count = process_tweet(tweet_text)
    hashtag_counter_seq.update(tags)

    if len(batch) >= next_workload_target:
        print(f"\n=======> Processing Workload: {next_workload_target} Tweets")

        # Sequential
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = psutil.virtual_memory().percent
        start_seq = time.time()

        hashtag_counter_seq.clear()
        for text in batch:
            words, tags, length, sentiment, word_count = process_tweet(text)
            hashtag_counter_seq.update(tags)

        elapsed_seq = time.time() - start_seq
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory().percent

        seq_metrics = {
            'time': round(elapsed_seq, 2),
            'cpu': round((cpu_before + cpu_after) / 2, 2),
            'mem': round((mem_before + mem_after) / 2, 2),
            'throughput': round(next_workload_target / elapsed_seq, 2),
            'top_tags': hashtag_counter_seq.most_common(TOP_N)
        }

        print(f"  Sequential | Time: {seq_metrics['time']}s | CPU: {seq_metrics['cpu']}% | Mem: {seq_metrics['mem']}% | Throughput: {seq_metrics['throughput']} rec/s")

        # Parallel
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = psutil.virtual_memory().percent
        start_par = time.time()

        hashtag_counter_par.clear()
        results = parallel_process(batch)
        for words, tags, length, sentiment, word_count in results:
            hashtag_counter_par.update(tags)

        elapsed_par = time.time() - start_par
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory().percent

        par_metrics = {
            'time': round(elapsed_par, 2),
            'cpu': round((cpu_before + cpu_after) / 2, 2),
            'mem': round((mem_before + mem_after) / 2, 2),
            'throughput': round(next_workload_target / elapsed_par, 2),
            'top_tags': hashtag_counter_par.most_common(TOP_N)
        }

        print(f"  Parallel   | Time: {par_metrics['time']}s | CPU: {par_metrics['cpu']}% | Mem: {par_metrics['mem']}% | Throughput: {par_metrics['throughput']} rec/s")

        results_per_workload.append({
            'workload': next_workload_target,
            'sequential': seq_metrics,
            'parallel': par_metrics
        })

        current_workload_index += 1
        if current_workload_index >= len(WORKLOADS):
            break

        next_workload_target = WORKLOADS[current_workload_index]
        batch.clear()
        window.clear()

consumer.close()

# ==== FINAL METRICS PRINT ====
print("\n=====> FINAL STREAM PROCESSING RESULTS\n")
for result in results_per_workload:
    print(f"\nWorkload: {result['workload']} Tweets")
    print(f"  Sequential:")
    print(f"    Time: {result['sequential']['time']}s | CPU: {result['sequential']['cpu']}% | Mem: {result['sequential']['mem']}% | Throughput: {result['sequential']['throughput']} rec/s")
    print("    Top Hashtags:")
    for tag, count in result['sequential']['top_tags']:
        print(f"      {tag}: {count}")

    print(f"  Parallel:")
    print(f"    Time: {result['parallel']['time']}s | CPU: {result['parallel']['cpu']}% | Mem: {result['parallel']['mem']}% | Throughput: {result['parallel']['throughput']} rec/s")
    print("    Top Hashtags:")
    for tag, count in result['parallel']['top_tags']:
        print(f"      {tag}: {count}")

# ==== PLOTTING METRICS ====
workloads = [r['workload'] for r in results_per_workload]
seq_times = [r['sequential']['time'] for r in results_per_workload]
par_times = [r['parallel']['time'] for r in results_per_workload]
seq_cpus = [r['sequential']['cpu'] for r in results_per_workload]
par_cpus = [r['parallel']['cpu'] for r in results_per_workload]
seq_mems = [r['sequential']['mem'] for r in results_per_workload]
par_mems = [r['parallel']['mem'] for r in results_per_workload]
seq_tps = [r['sequential']['throughput'] for r in results_per_workload]
par_tps = [r['parallel']['throughput'] for r in results_per_workload]

x = np.arange(len(workloads))
bar_width = 0.35

# Time Plot
plt.figure(figsize=(8,5))
plt.bar(x - bar_width/2, seq_times, bar_width, label='Sequential')
plt.bar(x + bar_width/2, par_times, bar_width, label='Parallel')
plt.xlabel('Workload (tweets)')
plt.ylabel('Time (s)')
plt.title('Processing Time by Workload')
plt.xticks(x, workloads)
plt.legend()
plt.tight_layout()
time_plot_path = os.path.join(RESULT_DIR, 'st_time_comparison.png')
plt.savefig(time_plot_path)
plt.close()
s3_client.upload_file(time_plot_path, BUCKET_NAME, 'graphs/st_time_comparison.png')

# CPU Plot
plt.figure(figsize=(8,5))
plt.bar(x - bar_width/2, seq_cpus, bar_width, label='Sequential')
plt.bar(x + bar_width/2, par_cpus, bar_width, label='Parallel')
plt.xlabel('Workload (tweets)')
plt.ylabel('CPU Usage (%)')
plt.title('CPU Usage by Workload')
plt.xticks(x, workloads)
plt.legend()
plt.tight_layout()
cpu_plot_path = os.path.join(RESULT_DIR, 'st_cpu_comparison.png')
plt.savefig(cpu_plot_path)
plt.close()
s3_client.upload_file(cpu_plot_path, BUCKET_NAME, 'graphs/st_cpu_comparison.png')

# Memory Plot
plt.figure(figsize=(8,5))
plt.bar(x - bar_width/2, seq_mems, bar_width, label='Sequential')
plt.bar(x + bar_width/2, par_mems, bar_width, label='Parallel')
plt.xlabel('Workload (tweets)')
plt.ylabel('Memory Usage (%)')
plt.title('Memory Usage by Workload')
plt.xticks(x, workloads)
plt.legend()
plt.tight_layout()
mem_plot_path = os.path.join(RESULT_DIR, 'st_memory_comparison.png')
plt.savefig(mem_plot_path)
plt.close()
s3_client.upload_file(mem_plot_path, BUCKET_NAME, 'graphs/st_memory_comparison.png')

# Throughput Plot
plt.figure(figsize=(8,5))
plt.bar(x - bar_width/2, seq_tps, bar_width, label='Sequential')
plt.bar(x + bar_width/2, par_tps, bar_width, label='Parallel')
plt.xlabel('Workload (tweets)')
plt.ylabel('Throughput (records/sec)')
plt.title('Throughput by Workload')
plt.xticks(x, workloads)
plt.legend()
plt.tight_layout()
tps_plot_path = os.path.join(RESULT_DIR, 'st_throughput_comparison.png')
plt.savefig(tps_plot_path)
plt.close()
s3_client.upload_file(tps_plot_path, BUCKET_NAME, 'graphs/st_throughput_comparison.png')

# Final overall sentiment, word count, length metrics
df = pd.read_csv(DATASET_PATH)
texts = df['text'].dropna().astype(str).tolist()

sentiments = []
word_counts = []
lengths = []

for text in texts:
    words = clean_and_tokenize(text)
    sentiments.append(get_sentiment(text))
    word_counts.append(len(words))
    lengths.append(len(text))

avg_sentiment = sum(sentiments) / len(sentiments)
avg_word_count = sum(word_counts) / len(word_counts)
avg_length = sum(lengths) / len(lengths)

print(f"\n=====> Overall Dataset Metrics:")
print(f"  Average Sentiment     : {avg_sentiment:.4f}")
print(f"  Average Word Count    : {avg_word_count:.2f}")
print(f"  Average Tweet Length  : {avg_length:.2f}")

print(f"\n=====> Plots saved to {RESULT_DIR}/ and uploaded to s3://{BUCKET_NAME}/graphs/\n")
print("\n=====> Stream processing ended cleanly.\n")
