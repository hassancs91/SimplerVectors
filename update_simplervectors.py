import numpy as np
import os
import pickle
import json
import h5py
import enum
from  vector_functions import cdist, _calculate_cosine_distances
#from scipy.spatial.distance import cdist, hamming


class SerializationFormat(enum.Enum):
    BINARY = 'pickle'
    JSON = 'json'
    HIERARCHICAL = 'hdf5'


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
        elif serialization_format == SerializationFormat.JSON:
            self._load_json(file_path)
        elif serialization_format == SerializationFormat.HIERARCHICAL:
            self._load_hdf5(file_path)

    def save_to_disk(self, collection_name, serialization_format=SerializationFormat.BINARY):
        file_path = os.path.join(self.db_folder, collection_name + '.svdb')
        if serialization_format == SerializationFormat.BINARY:
            self._save_pickle(file_path)
        elif serialization_format == SerializationFormat.JSON:
            self._save_json(file_path)
        elif serialization_format == SerializationFormat.HIERARCHICAL:
            self._save_hdf5(file_path)

    def _load_pickle(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.vectors, self.metadata = pickle.load(file)
        else:
            self.vectors, self.metadata = [], []

    def _save_pickle(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump((self.vectors, self.metadata), file)

    def _load_json(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
                self.vectors = [np.array(vec) for vec in data['vectors']]
                self.metadata = data['metadata']
        else:
            self.vectors, self.metadata = [], []

    def _save_json(self, file_path):
        data = {'vectors': [vec.tolist() for vec in self.vectors], 'metadata': self.metadata}
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def _load_hdf5(self, file_path):
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as file:
                self.vectors = [np.array(file['vectors'][str(i)]) for i in range(len(file['vectors']))]
                self.metadata = [json.loads(str(file['metadata'][str(i)])) for i in range(len(file['metadata']))]
        else:
            self.vectors, self.metadata = [], []

    def _save_hdf5(self, file_path):
        with h5py.File(file_path, 'w') as file:
            group_vectors = file.create_group('vectors')
            group_metadata = file.create_group('metadata')
            for i, (vec, meta) in enumerate(zip(self.vectors, self.metadata)):
                group_vectors.create_dataset(str(i), data=np.array(vec))
                group_metadata.create_dataset(str(i), data=json.dumps(meta))

    def load_from_disk_old(self, collection_name):
        file_path = os.path.join(self.db_folder, collection_name + '.svdb')
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                self.vectors, self.metadata = pickle.load(file)
        else:
            self.vectors, self.metadata = [], []

    def save_to_disk_old(self, collection_name):
        file_path = os.path.join(self.db_folder, collection_name + '.svdb')
        with open(file_path, 'wb') as file:
            pickle.dump((self.vectors, self.metadata), file)

    @staticmethod
    def normalize_vector(vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector  # Prevents division by zero
        return vector / norm

    def add_vector(self, vector, meta, normalize=False):
        if normalize:
            vector = self.normalize_vector(vector)
        self.vectors.append(vector)
        self.metadata.append(meta)

    def add_vectors_batch(self, vectors_with_meta, normalize=False):
        for vector, meta in vectors_with_meta:
            self.add_vector(vector, meta, normalize=normalize)

    def find_similar_with_cosine(self, vector, top_n=10):
        return self.find_similar(vector, top_n, 'cosine')

    def find_similar_with_euclidean(self, vector, top_n=10):
        return self.find_similar(vector, top_n, 'euclidean')

    def find_similar_with_manhattan(self, vector, top_n=10):
        return self.find_similar(vector, top_n, 'cityblock')



    # def find_similar_with_hamming_distance(self, vector, top_n=10):
    #     if not self.vectors:
    #         return []
    #     vectors_matrix = np.array(self.vectors)
    #     distances = np.array([hamming(vector, v) for v in vectors_matrix])
    #     top_indices = np.argsort(distances)[:top_n]
    #     return [(self.metadata[i], distances[i]) for i in top_indices]
    
    def find_similar(self, vector, top_n, metric):
        if not self.vectors:
            return []
        vectors_matrix = np.array(self.vectors)
        distances = cdist([vector], vectors_matrix, metric)
        if metric == 'cosine':
            distances = 1 - distances  # Cosine similarity is 1 - cosine distance
        top_indices = np.argsort(distances[0])[:top_n]
        return [(self.metadata[i], distances[0][i]) for i in top_indices]

    # def find_similar_old(self, vector, top_n=10):
    #     if not self.vectors:
    #         return []
    #     vectors_matrix = np.array(self.vectors)
    #     cosine_similarity = 1 - cdist([vector], vectors_matrix, 'cosine')
    #     top_indices = np.argsort(cosine_similarity[0])[-top_n:][::-1]
    #     return [(self.metadata[i], cosine_similarity[0][i]) for i in top_indices]

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

    def update_vectors_batch(self, key, value, new_vectors):
        for i, meta in enumerate(self.metadata):
            if meta.get(key) == value:
                try:
                    self.vectors[i] = new_vectors.pop(0)
                except IndexError:
                    break  # No more vectors to update

    def delete_vectors_batch(self, key, value):
        to_delete = [i for i, meta in enumerate(self.metadata) if meta.get(key) == value]
        for i in sorted(to_delete, reverse=True):
            del self.vectors[i]
            del self.metadata[i]

    def filter_by_metadata(self, criteria):
        """
        Filter vectors by multiple metadata criteria.
        Args:
            criteria (dict): A dictionary of metadata criteria.
        Returns:
            List of tuples (vector, metadata) that match all the criteria.
        """
        results = []
        for vec, meta in zip(self.vectors, self.metadata):
            if all(meta.get(k) == v for k, v in criteria.items()):
                results.append((vec, meta))
        return results

    def filter_by_metadata_range(self, key, min_val, max_val):
        """
        Filter vectors by a range of values for a numeric metadata key.
        Args:
            key (str): The metadata key.
            min_val (numeric): The minimum value for the range.
            max_val (numeric): The maximum value for the range.
        Returns:
            List of tuples (vector, metadata) where metadata[key] falls within the range.
        """
        results = []
        for vec, meta in zip(self.vectors, self.metadata):
            value = meta.get(key)
            if value is not None and min_val <= value <= max_val:
                results.append((vec, meta))
        return results




import numpy as np

def get_cosine_similarity_base(embeddings):
    # Manually calculate the cosine similarities between consecutive embeddings.
    similarities = []
    for i in range(len(embeddings) - 1):
        vec1 = embeddings[i].flatten()
        vec2 = embeddings[i + 1].flatten()
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            # If either vector is zero, similarity is undefined (could also return 0)
            similarity = float("nan")
        else:
            similarity = dot_product / (norm_vec1 * norm_vec2)
        similarities.append(similarity)
    return similarities

def calculate_cosine_distances(embeddings):
    # Correctly calculate the cosine distance between consecutive embeddings
    distances = []
    similarities = get_cosine_similarity(embeddings)
    for similarity in similarities:
        distance = 1 - similarity
        distances.append(distance)
    return distances

def get_cosine_similarity(vec1, vec2):
    # Calculate the cosine similarity between two individual vectors.
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        # If either vector is zero, similarity is undefined (could also return 0)
        return float("nan")
    else:
        return dot_product / (norm_vec1 * norm_vec2)
