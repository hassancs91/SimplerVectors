from embeddings import generate_embeddings_open_ai,generate_text
from SimplerVectors import VectorDatabase
from chunker import chunk_by_max_chunk_size

# file_path = "doc.txt"

# main_file = ""

# with open('doc.txt', 'r',encoding="utf-8") as file:
#     # Read the content of the file
#     main_file = file.read()


# documents = chunk_by_max_chunk_size(main_file,200, preserve_sentence_structure=True)



# chunks = documents.chunks

# text_chunks = [chunk.text for chunk in chunks]

# embedded_documents = []

# for chunk in text_chunks:
#     embedding = generate_embeddings_open_ai(chunk)
#     embedded_documents.append(embedding)



db = VectorDatabase('data')
# # Adding each vector to the database with some metadata (e.g., the document it came from)
# for idx, emb in enumerate(embedded_documents):
#     db.add_vector(emb, {"doc_id": idx, "vector": text_chunks[idx]}, normalize=True)


# db.save_to_disk("rag.db")

db.load_from_disk("rag.db")



# Example query
query = "how to define the functions for agent?"

# Embed the query using the same method as the documents
query_embedding = generate_embeddings_open_ai(query)
query_embedding = db.normalize_vector(query_embedding)  # Normalizing the query vector

# Retrieving the top similar document
results = db.top_cosine_similarity(query_embedding, top_n=1)

results_text= []
for doc, score in results:
     results_text.append(doc['vector'])


prompt = f"Answer the following question: {query} \n Based on this context only: \n" + results_text[0]

answer = generate_text(prompt)

print(answer)

# for doc, score in results:
#     print(f"Vector: {doc['vector']}, Similarity Score: {score}")


#






