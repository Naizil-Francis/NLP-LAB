import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import euclidean, cityblock
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Shipment of gold damaged in a fire",
    "Delivery of silver arrived in a silver truck",
    "Shipment of gold arrived in a truck",
    "Purchased silver and gold arrived in a wooden truck",
    "The arrival of gold and silver shipment is delayed."
]#Document set

query = "gold silver truck" #query sentence

vectorizer = CountVectorizer(stop_words="english")#initialise the tdf with no matrix
X = vectorizer.fit_transform(documents + [query])#push the document and query to tdf
vectors = X.toarray()#make it in array form
doc_vectors = vectors[:-1]#docs only
query_vector = vectors[-1]#query only

def compute_distances(doc_vectors, query_vector):#distance calculations
    euclidean_distances = [euclidean(doc, query_vector) for doc in doc_vectors]
    manhattan_distances = [cityblock(doc, query_vector) for doc in doc_vectors]
    cosine_similarities = cosine_similarity(doc_vectors, query_vector.reshape(1, -1)).flatten()
    return euclidean_distances, manhattan_distances, cosine_similarities

euclidean_distances, manhattan_distances, cosine_similarities = compute_distances(doc_vectors, query_vector)

euclidean_ranking = np.argsort(euclidean_distances)#more distance, unlikely related
manhattan_ranking = np.argsort(manhattan_distances)#more distance, unlikely related
cosine_ranking = np.argsort(-cosine_similarities)#more cosine value, likely related

#best 2 docs (+1 since indexing is from 0)
top_2_euclidean = euclidean_ranking[:2]+1
top_2_manhattan = manhattan_ranking[:2]+1
top_2_cosine = cosine_ranking[:2]+1

print("Euclidean Distance:",euclidean_distances)
print("Manhattan Distance:",manhattan_distances)
print("Cosine Similarity:",cosine_similarities)

print("\nTop 2 documents using Euclidean distance:", top_2_euclidean)
print("Top 2 documents using Manhattan distance:", top_2_manhattan)
print("Top 2 documents using Cosine similarity:", top_2_cosine)
