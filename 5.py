from collections import defaultdict, Counter
import math

reviews = [
    ("fun, couple, love, love", "comedy"),
    ("fast, furious, shoot", "action"),
    ("couple, fly, fast, fun, fun", "comedy"),
    ("furious, shoot, shoot, fun", "action"),
    ("fly, fast, shoot, love", "action")
]#define reviews

D = "fast, couple, shoot, fly" #the query

def tokenize(text):
    return text.split(", ") #split into words

class_docs = defaultdict(list)#tokens based on class
vocabulary = set()#vocab set (no repeatations).
class_count=defaultdict(int)

for review, catgory in reviews: #classifying into classes
    tokens = tokenize(review)#tokenisation
    class_docs[catgory].extend(tokens)#add tokens to respective class
    class_count[catgory] += 1 #increment the class count
    vocabulary.update(tokens)#update the vocab set

vocab_size = len(vocabulary)
total_docs = len(reviews)
priors = {catgory: count / total_docs for catgory,count in class_count.items()} #prior probability calculation for each class

likelihoods = {}
for catgory, tokens in class_docs.items(): #calculating likelihood probability for each class
    token_counts = Counter(tokens)
    total_words = len(tokens)
    likelihoods[catgory] = {word: (token_counts[word] + 1) / (total_words + vocab_size) for word in vocabulary} # laplace smoothing

tokens = tokenize(D)#tokenise the query document
posteriors = {}
    
for catgory in priors:#calculating posterior probability for each class
    log_prob = (priors[catgory])
    for token in tokens:
        log_prob *= (likelihoods[catgory].get(token, 1 / (len(class_docs[catgory]) + vocab_size)))
    posteriors[catgory] = log_prob

most_likely_class = max(posteriors, key=posteriors.get)
print('Posterior Probability:', posteriors)
print(f"The most likely class for the document '{D}' is: {most_likely_class}")