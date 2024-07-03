#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
using namespace std;

int main() {
    string corpus[10], test;
    int count;
    float probability = 1.0; //initial probability

    cout << "Enter the number of sentences in the corpus: ";
    cin >> count;
    cin.ignore();
    cout << "Enter the sentences for the corpus:" << endl;
    for (int i = 0; i < count; i++) {
        cout << "Sentence " << i+1 << ": ";
        getline(cin, corpus[i]);
        corpus[i] = "<s> " + corpus[i] + " </s>";// start and end with tags
    }

    cout << "Enter the test sentence: ";
    getline(cin, test);
    stringstream test_ss("<s> " + test + " </s>");

    string word;
    vector<string> test_words;// vector of words from test sentences
    while (test_ss >> word) test_words.push_back(word);

    vector<string> words;//vector of words for corpus
    map<string, int> unigram;//unigram
    map<pair<string, string>, int> bigram;//bigram
    for (int i = 0; i < count; i++) {
        stringstream ss(corpus[i]);
        string prev_word = "<s>", word;//previous tag ie beginning of sentence
        while (ss >> word) {
            words.push_back(word);
            unigram[word]++;//unigram
            bigram[{prev_word, word}]++;//bigram
            prev_word = word;//prev word for bigrams
        }
    }

    cout << "\nUnigram:" << endl;//display unigrams
    for (const auto& word : test_words)
        if (unigram.find(word) != unigram.end())
            cout << word << ": " << unigram[word] << endl;

    cout << "\nBigram:" << endl;//display bigrams
    for (int i = 0; i < test_words.size() - 1; i++) {
        auto current_bigram = make_pair(test_words[i], test_words[i + 1]);//current and next word pairing for test sentence
        if (bigram.find(current_bigram) != bigram.end())//traverse till end.
            cout << "(" << current_bigram.first << ", " << current_bigram.second << "): " << bigram[current_bigram] << endl;
    }

    for (int i = 0; i < test_words.size() - 1; i++) {//calculate probability
        if (bigram.find({test_words[i], test_words[i + 1]}) != bigram.end()) {
            probability *= (float)bigram[{test_words[i], test_words[i + 1]}] / unigram[test_words[i]];//probability calculation
        }
    }

    cout << "\nProbability:" <<probability<< endl;
    return 0;
}
