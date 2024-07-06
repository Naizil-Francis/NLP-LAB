import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import euclidean, jaccard
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "It is going to rain today",
    "Today Rama is not going outside to watch rain",
    "I am going to watch the movie tomorrow with Rama",
    "Tomorrow Rama is going to watch the rain at sea shore"
]

query = "Rama watching the rain"

vectorizer = TfidfVectorizer(stop_words='english')
X_docs = vectorizer.fit_transform(documents).toarray()
X_query = vectorizer.transform([query]).toarray()

lsa = TruncatedSVD(n_components=4)
X_docs_lsa = lsa.fit_transform(X_docs)
X_query_lsa = lsa.transform(X_query)

def compute_similarity_measures(doc_vectors, query_vector):
    euclidean_distances = [euclidean(doc, query_vector) for doc in doc_vectors]
    cosine_similarities = cosine_similarity(doc_vectors, query_vector.reshape(1, -1)).flatten()
    
    jaccard_similarities = []
    dice_similarities = []
    
    for doc in doc_vectors:
        doc_binary = np.array(doc > 0, dtype=int)
        query_binary = np.array(query_vector > 0, dtype=int)
        
        jaccard_sim = 1 - jaccard(doc_binary, query_binary)
        dice_sim = 2 * np.sum(doc_binary & query_binary) / (np.sum(doc_binary) + np.sum(query_binary))
        
        jaccard_similarities.append(jaccard_sim)
        dice_similarities.append(dice_sim)
    
    return euclidean_distances, cosine_similarities, jaccard_similarities, dice_similarities

euclidean_distances, cosine_similarities, jaccard_similarities, dice_similarities = compute_similarity_measures(X_docs_lsa, X_query_lsa[0])

euclidean_ranking = np.argsort(euclidean_distances)
cosine_ranking = np.argsort(-cosine_similarities)
jaccard_ranking = np.argsort(-np.array(jaccard_similarities))
dice_ranking = np.argsort(-np.array(dice_similarities))

def plot_rankings(distances, measure_names):
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'c']
    bar_width = 0.2
    positions = np.arange(len(distances[0]))

    for i, dist in enumerate(distances):
        plt.bar(positions + i * bar_width, dist , bar_width, label=measure_names[i], color=colors[i])

    plt.xlabel('Documents')
    plt.ylabel('Similarities')
    plt.title('Documents Comparison')
    plt.xticks(positions + bar_width, [f'D{i+1}' for i in range(len(distances[0]))])
    plt.legend()
    plt.tight_layout()
    plt.show()

print("Top 2 documents using Euclidean distance:", euclidean_ranking[:2] + 1)
print("Top 2 documents using Cosine similarity:", cosine_ranking[:2] + 1)
print("Top 2 documents using Jaccard similarity:", jaccard_ranking[:2] + 1)
print("Top 2 documents using Dice similarity coefficient:", dice_ranking[:2] + 1)

print("Euclidean distance:", euclidean_distances)
print("Cosine similarity:", cosine_similarities)
print("Jaccard similarity:", jaccard_similarities)
print("Dice similarity coefficient:", dice_similarities)

measure_names = ["Euclidean Distance", "Cosine Similarity", "Jaccard Similarity", "Dice Similarity"]
dist = [euclidean_distances, cosine_similarities, jaccard_similarities, dice_similarities]

plot_rankings(dist, measure_names)

