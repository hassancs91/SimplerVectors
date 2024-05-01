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
