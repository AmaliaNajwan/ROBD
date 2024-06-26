import pymongo
import requests
import time


client = pymongo.MongoClient("mongodb+srv://amalianajwa1193:rRawMjsbTqZTOTOf@cluster0.djkgm3b.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.sample_airbnb
collection = db.listingsAndReviews

hf_token = "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"
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

def generate_embedding(text: str) -> list[float]:

  response = requests.post(
    embedding_url,
    headers={"Authorization": f"Bearer {hf_token}"},
    json={"inputs": text})

  if response.status_code != 200:
    raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

  return response.json()

print("Test Embedding:", generate_embedding("This is a text for testing the embedding"))

#for doc in collection.find({'summary':{"$exists": True}}).limit(50):
#   doc['summary_embedding_hf'] = generate_embedding(doc['summary'])
#   collection.replace_one({'_id': doc['_id']}, doc)

# Optimize by caching embeddings for common queries
embedding_cache = {}

def search_listings_optimized(query: str, num_candidates: int = 150, limit: int = 4, max_price: float = None):
    if query in embedding_cache:
        query_embedding = embedding_cache[query]
    else:
        query_embedding = generate_embedding(query)
        embedding_cache[query] = query_embedding

    pipeline = [
        {"$vectorSearch": {
            "queryVector": query_embedding,
            "path": "summary_embedding_hf",
            "numCandidates": num_candidates,
            "limit": limit,
            "index": "vector_index",
        }}
    ]
    
    if max_price is not None:
        pipeline.append({"$match": {"price": {"$lt": max_price}}})

    results = collection.aggregate(pipeline)
    for document in results:
        print(f'AirBnB Name: {document["name"]},\nSummary: {document["summary"]}\n')

queries = [
    "scenic view with bathroom",
    "cozy and quiet place",
    "family friendly environment",
    "near public transport",
    "pet friendly accommodation",
    "luxury experience with amenities",
    "affordable budget stay"
]

for i, query in enumerate(queries, start=1):
    print(f"Optimized Query {i}: {query}")
    search_listings_optimized(query, max_price=200)  # Example filter for price < 200
    print("\n" + "="*50 + "\n")

# Measure and compare performance
def measure_performance(query_func, query: str, *args, **kwargs):
    start_time = time.time()
    query_func(query, *args, **kwargs)
    end_time = time.time()
    return end_time - start_time

for i, query in enumerate(queries, start=1):
    print(f"Measuring performance for Query {i}: {query}")
    original_time = measure_performance(search_listings_optimized, query, 100, 4)
    optimized_time = measure_performance(search_listings_optimized, query, 150, 4, 200)
    print(f"Original Query Time: {original_time:.2f} seconds")
    print(f"Optimized Query Time: {optimized_time:.2f} seconds")
    print("\n" + "="*50 + "\n")