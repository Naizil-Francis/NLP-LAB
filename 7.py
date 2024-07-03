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
]#document set

query = "Rama watching the rain" #query

vectorizer = TfidfVectorizer(stop_words='english') #create tdf
X_docs = vectorizer.fit_transform(documents).toarray()# fit the document in tdf and vectorise it
X_query = vectorizer.transform([query]).toarray()#vectorise query document from tdf

lsa = TruncatedSVD(n_components=2)# SVD defining
X_docs_lsa = lsa.fit_transform(X_docs)#fit doc to document tdf
X_query_lsa = lsa.transform(X_query)# fit query  to tdf

def compute_similarity_measures(doc_vectors, query_vector):#calculate similarity index
    euclidean_distances = [euclidean(doc, query_vector) for doc in doc_vectors]
    cosine_similarities = cosine_similarity(doc_vectors, query_vector.reshape(1, -1)).flatten()
    
    jaccard_similarities = []
    dice_similarities = []
    
    for doc in doc_vectors:#jaccard distance 
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

top_2_euclidean = euclidean_ranking[:2]
top_2_cosine = cosine_ranking[:2]
top_2_jaccard = jaccard_ranking[:2]
top_2_dice = dice_ranking[:2]

def plot_rankings(rankings, measure_names):
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r', 'c']  # Colors for each measure
    bar_width = 0.2  # Width of each bar
    positions = np.arange(len(rankings[0]))  # Positions for the bars

    for i, ranking in enumerate(rankings):
        plt.bar(positions + i * bar_width, ranking, bar_width, label=measure_names[i], color=colors[i])

    plt.xlabel('Documents')
    plt.ylabel('Ranking')
    plt.title('Top 2 Documents Comparison')
    plt.xticks(positions + bar_width, [f'D{i+1}' for i in range(len(rankings[0]))])
    plt.legend()
    plt.tight_layout()
    plt.show()

print("Top 2 documents using Euclidean distance:", top_2_euclidean)
print("Top 2 documents using Cosine similarity:", top_2_cosine)
print("Top 2 documents using Jaccard similarity:", top_2_jaccard)
print("Top 2 documents using Dice similarity coefficient:", top_2_dice)

measure_names = ["Euclidean Distance", "Cosine Similarity", "Jaccard Similarity", "Dice Similarity"]
rankings = [top_2_euclidean, top_2_cosine, top_2_jaccard, top_2_dice]

plot_rankings(rankings, measure_names)