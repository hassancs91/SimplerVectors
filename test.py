from SimplerVectors_core import VectorDatabase
from embeddings import get_embedding

# Sample document consisting of 10 sentences
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming industries.",
    "The Great Wall of China is visible from space.",
    "Python is a versatile programming language.",
    "The human brain is an intricate organ.",
    "Quantum computing will revolutionize technology.",
    "The Louvre is the world's largest art museum.",
    "Climate change is a critical global challenge.",
    "The Amazon rainforest is vital for global ecology.",
    "Blockchain technology underpins cryptocurrencies."
]

# Embedding all documents
embedded_documents = [get_embedding(doc) for doc in documents]


db = VectorDatabase('VDB')

# Adding each vector to the database with some metadata (e.g., the document it came from)
for idx, emb in enumerate(embedded_documents):
    db.add_vector(emb, {"doc_id": idx, "vector": documents[idx]}, normalize=True)


# Save the database to disk (choose a collection name, e.g., "test_collection")
db.save_to_disk("test_json")


# Example query
query = "Quantum computing is transforming industries"

# Embed the query using the same method as the documents
query_embedding = get_embedding(query)
query_embedding = db.normalize_vector(query_embedding)  # Normalizing the query vector

# Retrieving the top similar document
results = db.top_cosine_similarity(query_embedding, top_n=3)
for doc, score in results:
    print(f"Vector: {doc['vector']}, Similarity Score: {score}")
#print("Most relevant sentence:", results)



