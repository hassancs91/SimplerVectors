from transformers import AutoTokenizer, AutoModel
import SimplerVectors

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

# Function to generate embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    # Use .squeeze() to convert to 2D array if necessary
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

# Example sentences
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "I love pizza",
    "The Eiffel Tower is located in Paris",
    "Machine learning is fascinating",
    "The weather today is sunny",
    "I enjoy reading books",
    "My favorite color is blue",
    "Music brings joy to many people",
    "Python is a popular programming language",
    "Artificial Intelligence will shape the future"
]

# Initialize the database
db = SimplerVectors.VectorDatabase('my_db')

# Add embeddings of sentences to the database
# for idx, sentence in enumerate(sentences):
#     embedding = get_embedding(sentence)
#     db.add_vector(embedding, {'id': idx, 'sentence': sentence})

# Save the database to disk
#db.save_to_disk('test_embeddings.svdb')

# Load the database
db.load_from_disk('test_embeddings.svdb')

# Find vectors similar to the embedding of a query sentence
query_sentence = "I enjoy sunny weather"
query_embedding = get_embedding(query_sentence)
similar_sentences = db.find_similar_with_euclidean(query_embedding, top_n=3)

for meta, score in similar_sentences:
    sentence = meta['sentence']  # Correctly accessing the sentence from the metadata
    print(f"Sentence: {sentence}, Similarity: {score}")
