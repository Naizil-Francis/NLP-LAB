1. Consider the following Corpus of three sentences and Calculate P for the sentence “They play in a big Garden” assuming a bi-gram language model.
	a) There is a big garden. 
	b) Children play in a garden 
    c) They play inside beautiful garden

2. Find the bigram count for the given corpus. Apply Laplace smoothing and find the bigram probabilities after add-one smoothing (up to 4 decimal places)

3. Implement rule-based tagger and stochastic tagger for the give corpus of sentences.

4. Implement top-down and bottom-up parsing using python NLTK.

5. Given the following short movie reviews, each labeled with a genre, either comedy or action:
   		a) fun, couple, love, love : comedy
   		b) fast, furious, shoot : action
   		c) couple, fly, fast, fun, fun :comedy
   		d) furious, shoot, shoot, fun :action
   		e) fly, fast, shoot, love :action
   A new document { D: fast, couple, shoot, fly }.
   Compute the most likely class for D. Assume a naive Bayes classifier and use add-1 smoothing for the likelihoods.

6. The dataset contains following 5 documents.
   		D1: "Shipment of gold damaged in a fire"
		D2: "Delivery of silver arrived in a silver truck"
		D3: "Shipment of gold arrived in a truck"
		D4: “Purchased silver and gold arrived in a wooden truck”
		D5: “The arrival of gold and silver shipment is delayed.”
	Find the top two relevant documents for the query document with the content “gold silver truck " using the vector space model. Use the following similarity 		measure and analyze the result.

7. The dataset contains following 4 documents.
   		D1: " It is going to rain today "
   		D2: " Today Rama is not going outside to watch rain"
   		D3: “I am going to watch the movie tomorrow with Rama"
   		D4: “Tomorrow Rama is going to watch the rain at sea shore "
   Find the top two relevant documents for the query document with the content “Rama watching the rain " using the latent semantic space model. Use the following 
   similarity measure and show the result analysis using bar chart.
	   a) Euclidean distance
	   b) Cosine similarity
	   c) Jaccard similarity
	   d) Dice Similarity Coefficient.

8. Extract Synonyms and Antonyms for a given word using WordNet.
   
9. Implement a machine translator for 10 words using encoder-decoder model for any two languages.
