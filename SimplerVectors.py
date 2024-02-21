import numpy as np
import os
import pickle
from scipy.spatial.distance import cdist

class VectorDatabase:
    def __init__(self, db_folder):
        self.db_folder = db_folder
        self.vectors = []  # Initialize the vectors list
        self.metadata = [] # Initialize the metadata list
        if not os.path.exists(self.db_folder):
            os.makedirs(self.db_folder)

    def load_from_disk(self, collection_name):
        file_path = os.path.join(self.db_folder, collection_name + '.svdb')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.vectors, self.metadata = pickle.load(file)
        else:
            self.vectors, self.metadata = [], []

    def save_to_disk(self, collection_name):
        file_path = os.path.join(self.db_folder, collection_name + '.svdb')
        with open(file_path, 'wb') as file:
            pickle.dump((self.vectors, self.metadata), file)

    def add_vector(self, vector, meta):
        self.vectors.append(vector)
        self.metadata.append(meta)

    def find_similar(self, vector, top_n=10):
        if not self.vectors:
            return []
        vectors_matrix = np.array(self.vectors)
        cosine_similarity = 1 - cdist([vector], vectors_matrix, 'cosine')
        top_indices = np.argsort(cosine_similarity[0])[-top_n:][::-1]
        return [(self.metadata[i], cosine_similarity[0][i]) for i in top_indices]

    def delete_vector_by_meta(self, key, value):
        to_delete = [i for i, meta in enumerate(self.metadata) if meta.get(key) == value]
        for i in sorted(to_delete, reverse=True):
            del self.vectors[i]
            del self.metadata[i]

    def delete_vectors_in_range(self, start, end):
        del self.vectors[start:end]
        del self.metadata[start:end]

    def update_vector(self, key, value, new_vector):
        """
        Update vectors whose metadata matches the given key-value pair.
        Args:
            key: Metadata key to match.
            value: Metadata value to match.
            new_vector: New vector to replace the old one.
        """
        for i, meta in enumerate(self.metadata):
            if meta.get(key) == value:
                self.vectors[i] = new_vector

    def select_by_meta(self, key, value):
        """
        Select and return vectors and metadata that match a given key-value pair.
        Args:
            key: Metadata key to match.
            value: Metadata value to match.
        Returns:
            List of tuples (vector, metadata) that match the criteria.
        """
        return [(vec, meta) for vec, meta in zip(self.vectors, self.metadata) if meta.get(key) == value]

    def update_vectors_in_range_by_meta(self, key, value, start, end, new_vector):
        """
        Update a range of vectors whose metadata matches the given key-value pair.
        Args:
            key: Metadata key to match.
            value: Metadata value to match.
            start: Starting index of the range.
            end: Ending index of the range.
            new_vector: New vector to replace the old ones in the specified range.
        """
        for i in range(start, min(end, len(self.vectors))):
            if self.metadata[i].get(key) == value:
                self.vectors[i] = new_vector
# Example usage
# db = VectorDatabase('vd.db')

# # Add some vectors to the database
# vectors_to_add = [
#     (np.array([0.1, 0.2, 0.3]), {'id': 1, 'description': 'First vector'}),
#     (np.array([0.4, 0.1, 0.6]), {'id': 2, 'description': 'Second vector'}),
#     (np.array([0.7, 0.8, 0.9]), {'id': 3, 'description': 'Third vector'}),
#     (np.array([0.1, 0.5, 0.9]), {'id': 4, 'description': 'Fourth vector'}),
#     (np.array([0.9, 0.4, 0.2]), {'id': 5, 'description': 'Fifth vector'}),
#     (np.array([0.3, 0.6, 0.1]), {'id': 6, 'description': 'Sixth vector'}),
#     (np.array([0.8, 0.8, 0.2]), {'id': 7, 'description': 'Seventh vector'}),
#     (np.array([0.5, 0.4, 0.6]), {'id': 8, 'description': 'Eighth vector'}),
#     (np.array([0.2, 0.3, 0.7]), {'id': 9, 'description': 'Ninth vector'}),
#     (np.array([0.4, 0.4, 0.4]), {'id': 10, 'description': 'Tenth vector'}),
# ]

# for vector, meta in vectors_to_add:
#     db.add_vector(vector, meta)

# # Save the database to disk
# db.save_to_disk()

# # Let's simulate a scenario where we load the database from disk
# db = VectorDatabase('vd.db')
# db.load_from_disk()

# # Now, let's find vectors similar to a new query vector
# query_vector = np.array([0.5, 0.5, 0.5])
# similar_vectors = db.find_similar_advanced(query_vector, top_n=5)

# print(similar_vectors)
