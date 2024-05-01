def calculate_cosine_similarities_manual(embeddings):
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


def _calculate_cosine_distances(embeddings):
    # Calculate the cosine distance (1 - cosine similarity) between consecutive embeddings.
    distances = []
    for i in range(len(embeddings) - 1):
        similarity = calculate_cosine_similarities_manual(
            [embeddings[i]], [embeddings[i + 1]]
        )[0][0]
        distance = 1 - similarity
        distances.append(distance)
    return distances
