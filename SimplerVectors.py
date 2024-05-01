import numpy as np
import os
import pickle
import enum


class SerializationFormat(enum.Enum):
    BINARY = 'pickle'

class VectorDatabase:
    def __init__(self, db_folder):
        self.db_folder = db_folder
        self.vectors = []  # Initialize the vectors list
        self.metadata = [] # Initialize the metadata list
        if not os.path.exists(self.db_folder):
            os.makedirs(self.db_folder)

    def load_from_disk(self, collection_name, serialization_format=SerializationFormat.BINARY):
        file_path = os.path.join(self.db_folder, collection_name + '.svdb')
        if serialization_format == SerializationFormat.BINARY:
            self._load_pickle(file_path)

    def save_to_disk(self, collection_name, serialization_format=SerializationFormat.BINARY):
        file_path = os.path.join(self.db_folder, collection_name + '.svdb')
        if serialization_format == SerializationFormat.BINARY:
            self._save_pickle(file_path)

    def _load_pickle(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.vectors, self.metadata = pickle.load(file)
        else:
            self.vectors, self.metadata = [], []

    def _save_pickle(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump((self.vectors, self.metadata), file)

    @staticmethod
    def normalize_vector(vector):
        """
        Normalize a vector to unit length; return the original vector if it is zero-length.

        Parameters:
            vector (array-like): The vector to be normalized.

        Returns:
            array-like: A normalized vector with unit length.
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector  # Handle zero-length vector to avoid division by zero
        return vector / norm

    def add_vector(self, vector, meta, normalize=True):
        if normalize:
            vector = self.normalize_vector(vector)
        self.vectors.append(vector)
        self.metadata.append(meta)

    def add_vectors_batch(self, vectors_with_meta, normalize=False):
        for vector, meta in vectors_with_meta:
            self.add_vector(vector, meta, normalize=normalize)

    def top_cosine_similarity(self, target_vector, top_n=5):
        """
        Calculate the cosine similarity between a target vector (assumed to be normalized) and each vector in the pre-normalized matrix,
        then return the indices of the top N most similar vectors along with their metadata.

        Parameters:
            target_vector (array-like): The normalized vector to compare against the matrix.
            top_n (int): The number of top indices to return.

        Returns:
            list: Tuples of metadata and similarity score for the top N most similar vectors.
        """
        try:
            # Calculate cosine similarities directly as dot products with normalized vectors
            similarities = np.dot(self.vectors, target_vector)
            
            # Get the indices of the top N similar vectors
            top_indices = np.argsort(-similarities)[:top_n]
            
            # Return metadata and similarity for the top N entries
            return [(self.metadata[i], similarities[i]) for i in top_indices]
        except Exception as e:
            print(f"An error occurred: {e}")
            return []





