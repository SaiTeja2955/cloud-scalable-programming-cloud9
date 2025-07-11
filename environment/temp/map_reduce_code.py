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
from wordcloud import WordCloud
import seaborn as sns


# Config
BUCKET_NAME = 'x23411520-s3-bucket'
DATASET_KEY = 'covid-tweets-dataset/covidvaccine.csv'
DATASET_PATH = '/home/ec2-user/environment/x23418613-c9-env/environment/Dataset/covidvaccine.csv'
RESULT_DIR = '/home/ec2-user/environment/x23418613-c9-env/environment/temp'
TOP_N = 10
os.makedirs(RESULT_DIR, exist_ok=True)

# S3 Client
s3_client = boto3.client('s3')

#Download dataset from S3 only if not present locally
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


def process_tweet(text):
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

    s3_client.upload_file(local_graph_path, BUCKET_NAME, f"graphs/{metric}_comparison.png")
    print(f"=> Uploaded {metric} graph to s3://{BUCKET_NAME}/graphs/{metric}_comparison.png")


# === WordCloud of full dataset ===
print("\n=====> Generating WordCloud for full dataset")
all_words = []
for text in texts:
    all_words.extend(clean_and_tokenize(text))

wordcloud_text = ' '.join(all_words)
wordcloud = WordCloud(width=1200, height=600, background_color='white').generate(wordcloud_text)
wordcloud_path = f"{RESULT_DIR}/wordcloud.png"
wordcloud.to_file(wordcloud_path)
print(f"=> WordCloud saved to {wordcloud_path}")
s3_client.upload_file(wordcloud_path, BUCKET_NAME, "graphs/wordcloud.png")
print(f"=> Uploaded WordCloud to s3://{BUCKET_NAME}/graphs/wordcloud.png")


# === Top 10 Hashtags Bar Chart ===
print("\n=====> Generating Top 10 Hashtags Bar Chart")
hashtag_counter = Counter()
for text in texts:
    tags = extract_hashtags_mentions(text)
    hashtags = [tag for tag in tags if tag.startswith('#')]
    hashtag_counter.update(hashtags)

top_hashtags = hashtag_counter.most_common(TOP_N)
if top_hashtags:
    hashtags, counts = zip(*top_hashtags)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(hashtags), y=list(counts), palette='viridis')
    plt.title("Top 10 Hashtags", fontsize=16)
    plt.xlabel("Hashtag", fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    hashtag_chart_path = f"{RESULT_DIR}/top10_hashtags.png"
    plt.savefig(hashtag_chart_path)
    plt.close()
    print(f"=> Top 10 Hashtags chart saved to {hashtag_chart_path}")
    s3_client.upload_file(hashtag_chart_path, BUCKET_NAME, "graphs/top10_hashtags.png")
    print(f"=> Uploaded to s3://{BUCKET_NAME}/graphs/top10_hashtags.png")
else:
    print("=> No hashtags found in dataset.")


# === Sentiment Distribution Bar Chart ===
print("\n=====> Generating Sentiment Distribution Chart")
sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
for text in texts:
    polarity = get_sentiment(text)
    if polarity > 0.1:
        sentiment_counts['Positive'] += 1
    elif polarity < -0.1:
        sentiment_counts['Negative'] += 1
    else:
        sentiment_counts['Neutral'] += 1

plt.figure(figsize=(8, 5))
sns.barplot(x=list(sentiment_counts.keys()), y=list(sentiment_counts.values()), palette='pastel')
plt.title("Sentiment Distribution", fontsize=16)
plt.xlabel("Sentiment", fontsize=13)
plt.ylabel("Count", fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()

sentiment_chart_path = f"{RESULT_DIR}/sentiment_distribution.png"
plt.savefig(sentiment_chart_path)
plt.close()
print(f"=> Sentiment Distribution chart saved to {sentiment_chart_path}")
s3_client.upload_file(sentiment_chart_path, BUCKET_NAME, "graphs/sentiment_distribution.png")
print(f"=> Uploaded to s3://{BUCKET_NAME}/graphs/sentiment_distribution.png")


# === Final Result Printout ===
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

print("\nAll workloads processed, graphs uploaded, visualizations done and results printed.")
