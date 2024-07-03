#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
using namespace std;

int main() {
    int count;
    cout << "Enter the number of sentences in the corpus: ";
    cin >> count;
    cin.ignore();

    vector<string> corpus(count);//dataset (corpus)
    cout << "Enter the sentences for the corpus:" << endl;
    for (int i = 0; i < count; i++) {
        cout << "Sentence " << i + 1 << ": ";
        getline(cin, corpus[i]);
    }

    cout << "Enter the test sentence: ";
    string test;//test sentence
    getline(cin, test);

    stringstream test_ss(test);
    vector<string> test_words;//set of words in test sentence
    string word;
    while (test_ss >> word) test_words.push_back(word);//tokenisation

    map<string, int> unigram;
    map<pair<string, string>, int> bigram;

    for (const auto& sentence : corpus) {
        stringstream ss(sentence);
        string prev_word;
        ss >> prev_word; // Read the first word
        unigram[prev_word]++; // Count the first word of each sentence
        string current_word;
        while (ss >> current_word) {
            unigram[current_word]++;//unigram addition
            bigram[{prev_word, current_word}]++;//bigram addition
            prev_word = current_word;//prev word for the bigram
        }
    }

    // Compute vocab size
    int vocab_size = unigram.size();

    // Display unigram counts
    cout << "\nUnigram Counts:" << endl;
    for (const auto& entry : unigram) {
        cout << entry.first << ": " << entry.second << endl;
    }

    // Display bigram counts
    cout << "\nBigram Counts:" << endl;
    for (const auto& entry : bigram) {
        cout << "(" << entry.first.first << ", " << entry.first.second << "): " << entry.second << endl;
    }

    // Compute probabilities for test sentence
    float probability = 1.0;
    cout << "\nBigram Probabilities:" << endl;
    for (size_t i = 0; i < test_words.size() - 1; i++) {
        string w1 = test_words[i];
        string w2 = test_words[i + 1];
        int bigram_count = bigram[{w1, w2}];
        int unigram_count = unigram[w1];
        float bigram_probability = (float)(bigram_count + 1) / (unigram_count + vocab_size);
        probability *= bigram_probability;
        cout << "(" << w1 << ", " << w2 << "): " << bigram_probability << endl;
    }
    cout << "\nOverall Probability: " << probability << endl;
    return 0;
}
