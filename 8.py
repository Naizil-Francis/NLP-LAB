import nltk
from nltk.corpus import wordnet

# Download WordNet data
nltk.download('wordnet')

def get_synonyms_antonyms(word):
    synonyms = set()
    antonyms = set()
    
    for syn in wordnet.synsets(word):# for all words in synset
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())#append synonyms
            if lemma.antonyms():#if antonym exist
                for ant in lemma.antonyms():
                    antonyms.add(ant.name())#append all antonyms
    
    return synonyms, antonyms

word = input("Enter the word to get synonym and antonym: ")
synonyms, antonyms = get_synonyms_antonyms(word)
print(f"Synonyms of '{word}': {synonyms}")
print(f"Antonyms of '{word}': {antonyms}")
