def k_means_clustering(points, k, initial_centroids, max_iterations):
    """
    K-means clustering (Lloyd's algorithm) starting from the given centroids.

    Parameters
    ----------
    points : list of tuples, e.g. [(1, 2), (1, 4), ...]
    k : int, number of clusters
    initial_centroids : list of k tuples used as the starting centroids
    max_iterations : int, maximum number of assignment/update rounds

    Returns
    -------
    list of k tuples: the final centroids, each coordinate rounded to 4 decimals
    """
    points = [tuple(float(c) for c in p) for p in points]
    centroids = [tuple(float(c) for c in c0) for c0 in initial_centroids[:k]]
    dim = len(points[0])

    def sq_dist(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    for _ in range(max_iterations):
        # Assignment step: each point joins its nearest centroid
        clusters = [[] for _ in range(k)]
        for p in points:
            nearest = min(range(k), key=lambda j: sq_dist(p, centroids[j]))
            clusters[nearest].append(p)

        # Update step: move each centroid to the mean of its cluster
        new_centroids = []
        for j in range(k):
            if clusters[j]:
                n = len(clusters[j])
                new_centroids.append(
                    tuple(sum(p[d] for p in clusters[j]) / n for d in range(dim))
                )
            else:
                new_centroids.append(centroids[j])  # empty cluster: keep old centroid

        if new_centroids == centroids:  # converged
            break
        centroids = new_centroids

    return [tuple(round(c, 4) for c in centroid) for centroid in centroids]


