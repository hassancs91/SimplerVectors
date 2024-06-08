import csv
from SimplerVectors_core import VectorDatabase
from embeddings import get_embedding

# Load Q&A data from a CSV file using the csv module
# questions = []
# answers = []

# with open('QandA.csv', mode='r', encoding='utf-8') as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         questions.append(row['Question'])
#         answers.append(row['Answer'])

# # Embedding all questions
# embedded_questions = [get_embedding(question) for question in questions]

# # Initialize the vector database
db = VectorDatabase('QA_DB')

# Adding each vector to the database with metadata including both question and answer
# for idx, emb in enumerate(embedded_questions):
#     db.add_vector(emb, {"doc_id": idx, "question": questions[idx], "answer": answers[idx]}, normalize=True)

# Save the database to disk
#db.save_to_disk("qa_collection")
db.load_from_disk("qa_collection")
# Example query: looking for a question related to a specific topic
# query = "what is a banana ?"

# # Embed the query using the same method as the questions
# query_embedding = get_embedding(query)
# query_embedding = db.normalize_vector(query_embedding)  # Normalizing the query vector

# # Retrieving the top similar questions and their answers
# results = db.top_cosine_similarity(query_embedding, top_n=1)
# for doc, score in results:
#     print(f"Question: {doc['question']}, Answer: {doc['answer']}, Similarity Score: {score}")

print("Hello! Ask me any question or type 'exit' to leave:")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    # Embed the user query
    query_embedding = get_embedding(user_input)
    query_embedding = db.normalize_vector(query_embedding)  # Normalizing the query vector

    # Retrieving the top similar questions and their answers
    results = db.top_cosine_similarity(query_embedding, top_n=1)
    if results:
        top_match = results[0]
        print(f"Bot: {top_match[0]['answer']} (Similarity: {top_match[1]:.2f})")
    else:
        print("Bot: I'm not sure how to answer that.")