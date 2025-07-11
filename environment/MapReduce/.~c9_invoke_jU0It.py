import pandas as pd
from collections import Counter
from multiprocessing import cpu_count, Pool
import time
import psutil
import os
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob
import boto3
import os


# Config
BUCKET_NAME = 'x23418613-s3-bucket'
DATASET_KEY = 'covid-tweets-dataset/covidvaccine.csv'
DATASET_PATH = '/home/ec2-user/environment/Dataset/covidvaccine.csv'
RESULT_DIR = '/home/ec2-user/environment/results'
TOP_N = 10
os.makedirs(RESULT_DIR, exist_ok=True)

# S3 Client
s3_client = boto3.client('s3')

# Download dataset from S3 only if not present locally
if not os.path.exists(DATASET_PATH):
    print(f"Dataset not found locally. Downloading from s3://{BUCKET_NAME}/{DATASET_KEY} ...")
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    s3_client.download_file(BUCKET_NAME, DATASET_KEY, DATASET_PATH)
    print("Download complete.")
else:
    print(f"Dataset already exists at {DATASET_PATH}, skipping download.")

# Load dataset
df = pd.read_csv(DATASET_PATH)


texts = df['text'].astype(str).tolist()

# Workloads to test
workloads = [10000, 100000, 200000]


def clean_and_tokenize(text):
    import re
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return words


def extract_hashtags_mentions(text):
    words = text.split()
    tags = [word for word in words if word.startswith('#') or word.startswith('@')]
    return tags


def get_sentiment(text):
    return TextBlob(text).sentiment.polarity


def process_tweet(text): # mapper
    words = clean_and_tokenize(text)
    tags = extract_hashtags_mentions(text)
    length = len(text)
    sentiment = get_sentiment(text)
    word_count = len(words)
    return words, tags, length, sentiment, word_count


def sequential_processing(data):
    start_time = time.time()
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().percent

    word_counter, tag_counter = Counter(), Counter()
    sentiments, word_counts, lengths = [], [], []

    for text in data:
        words, tags, length, sentiment, word_count = process_tweet(text)
        word_counter.update(words)
        tag_counter.update(tags)
        sentiments.append(sentiment)
        word_counts.append(word_count)
        lengths.append(length)

    elapsed_time = time.time() - start_time
    avg_sentiment = sum(sentiments) / len(sentiments)
    avg_word_count = sum(word_counts) / len(word_counts)
    avg_length = sum(lengths) / len(lengths)

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory().percent

    metrics = {
        'time': elapsed_time,
        'cpu': (cpu_before + cpu_after) / 2,
        'mem': (mem_before + mem_after) / 2,
        'avg_sentiment': avg_sentiment,
        'avg_word_count': avg_word_count,
        'avg_tweet_length': avg_length,
        'top_words': word_counter.most_common(TOP_N),
        'top_tags': tag_counter.most_common(TOP_N)
    }

    return metrics


def parallel_processing(data):
    start_time = time.time()
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().percent

    word_counter, tag_counter = Counter(), Counter()
    sentiments, word_counts, lengths = [], [], []

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_tweet, data, chunksize=500)

    for words, tags, length, sentiment, word_count in results:
        word_counter.update(words)
        tag_counter.update(tags)
        sentiments.append(sentiment)
        word_counts.append(word_count)
        lengths.append(length)

    elapsed_time = time.time() - start_time
    avg_sentiment = sum(sentiments) / len(sentiments)
    avg_word_count = sum(word_counts) / len(word_counts)
    avg_length = sum(lengths) / len(lengths)

    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory().percent

    metrics = {
        'time': elapsed_time,
        'cpu': (cpu_before + cpu_after) / 2,
        'mem': (mem_before + mem_after) / 2,
        'avg_sentiment': avg_sentiment,
        'avg_word_count': avg_word_count,
        'avg_tweet_length': avg_length,
        'top_words': word_counter.most_common(TOP_N),
        'top_tags': tag_counter.most_common(TOP_N)
    }

    return metrics


# Collect metrics
seq_metrics, par_metrics = [], []

print("\n=====> Sequential Processing Started")
for workload in workloads:
    data = texts[:workload]
    s_metric = sequential_processing(data)
    seq_metrics.append({'workload': workload, **s_metric})
    print(f"Sequential | {workload} tweets: Time: {s_metric['time']:.2f}s | CPU: {s_metric['cpu']:.2f}% | Mem: {s_metric['mem']:.2f}%")

print("\n=====> Parallel Processing Started")
for workload in workloads:
    data = texts[:workload]
    p_metric = parallel_processing(data)
    par_metrics.append({'workload': workload, **p_metric})
    print(f"Parallel   | {workload} tweets: Time: {p_metric['time']:.2f}s | CPU: {p_metric['cpu']:.2f}% | Mem: {p_metric['mem']:.2f}%")


# Plot time, cpu, mem graphs and upload to S3
metrics_to_plot = ['time', 'cpu', 'mem']
seq_df = pd.DataFrame(seq_metrics)
par_df = pd.DataFrame(par_metrics)

colors = {'Sequential': '#4C72B0', 'Parallel': '#DD8452'}

for metric in metrics_to_plot:
    plt.figure(figsize=(9, 5))
    bar_width = 0.35
    x = np.arange(len(workloads))

    plt.bar(x - bar_width/2, seq_df[metric], width=bar_width, label='Sequential', color=colors['Sequential'])
    plt.bar(x + bar_width/2, par_df[metric], width=bar_width, label='Parallel', color=colors['Parallel'], alpha=0.9)

    plt.title(f"{metric.capitalize()} Comparison", fontsize=16)
    plt.xlabel("Workload (tweets)", fontsize=13)
    plt.ylabel(metric.capitalize(), fontsize=13)
    plt.xticks(x, workloads)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    local_graph_path = f"{RESULT_DIR}/{metric}_comparison.png"
    plt.savefig(local_graph_path)
    plt.close()

    # Upload to S3
    s3_graph_key = f"graphs/{metric}_comparison.png"
    s3_client.upload_file(local_graph_path, BUCKET_NAME, s3_graph_key)
    print(f"=> Uploaded {metric} graph to s3://{BUCKET_NAME}/{s3_graph_key}")

print("\n\n=====> Final Results for All Workloads")

for i, workload in enumerate(workloads):
    print(f"\n===== Workload: {workload} tweets =====")
    print(f"Sequential:")
    print(f"  Avg Sentiment     : {seq_metrics[i]['avg_sentiment']:.4f}")
    print(f"  Avg Word Count    : {seq_metrics[i]['avg_word_count']:.2f}")
    print(f"  Avg Tweet Length  : {seq_metrics[i]['avg_tweet_length']:.2f}")
    print(f"  Top {TOP_N} Words : {seq_metrics[i]['top_words']}")
    print(f"  Top {TOP_N} Tags  : {seq_metrics[i]['top_tags']}")

    print(f"\nParallel:")
    print(f"  Avg Sentiment     : {par_metrics[i]['avg_sentiment']:.4f}")
    print(f"  Avg Word Count    : {par_metrics[i]['avg_word_count']:.2f}")
    print(f"  Avg Tweet Length  : {par_metrics[i]['avg_tweet_length']:.2f}")
    print(f"  Top {TOP_N} Words : {par_metrics[i]['top_words']}")
    print(f"  Top {TOP_N} Tags  : {par_metrics[i]['top_tags']}")

print("\nAll workloads processed, graphs uploaded and results printed.")
