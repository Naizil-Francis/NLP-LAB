#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>

using namespace std;

typedef unordered_map<string, int> StringIntMap;//unigram map
typedef vector<pair<string, string>> TaggedSentence;//bigram map

struct NaiveBayesModel {//define naive bayes model
    StringIntMap tagCounts;
    unordered_map<string, StringIntMap> wordTagCounts;
    int totalTags;
};

void trainNaiveBayes(NaiveBayesModel& model, const vector<TaggedSentence>& taggedSentences) {//training function
    model.totalTags = 0;//initial tags =0
    for (const auto& sentence : taggedSentences) {
        for (const auto& wordTagPair : sentence) {
            const string& word = wordTagPair.first;//word
            const string& tag = wordTagPair.second;//tag
            model.tagCounts[tag]++;//tag increment
            model.wordTagCounts[tag][word]++;//word with tag occurence
            model.totalTags++;//total tags increment
        }
    }
}

vector<string> tagSentenceNaiveBayes(const NaiveBayesModel& model, const vector<string>& sentence) {//testing function
    vector<string> tags(sentence.size());
    for (size_t i = 0; i < sentence.size(); ++i) {
        const string& word = sentence[i];
        double maxProb = -1;//initially -1(invalid)
        string bestTag;
        for (const auto& tagCount : model.tagCounts) {
            const string& tag = tagCount.first;
            int wordCount = 0;
            auto tagIt = model.wordTagCounts.find(tag);//get tag
            if (tagIt != model.wordTagCounts.end()) {//best match or not 
                auto wordIt = tagIt->second.find(word);
                if (wordIt != tagIt->second.end()) {
                    wordCount = wordIt->second;
                }
            }
            double wordGivenTagProb = static_cast<double>(wordCount + 1) / 
                                      (tagCount.second + model.wordTagCounts.at(tag).size());//p(word|tag)
            double tagProb = static_cast<double>(tagCount.second) / model.totalTags;
            double prob = wordGivenTagProb * tagProb;//multiply
            if (prob > maxProb) {//if better, then update
                maxProb = prob;
                bestTag = tag;
            }
        }
        tags[i] = bestTag;
    }
    return tags;
}

int main() {
    vector<TaggedSentence> taggedSentences = {//dataset
        {{"John", "NNP"}, {"saw", "VBD"}, {"the", "DT"}, {"dog", "NN"}},
        {{"Mary", "NNP"}, {"walked", "VBD"}, {"to", "TO"}, {"the", "DT"}, {"park", "NN"}}
    };

    NaiveBayesModel model;//create model
    trainNaiveBayes(model, taggedSentences);//training phase

    vector<string> sentence = {"John", "walked", "to", "the", "park"};//test sentence
    vector<string> tags = tagSentenceNaiveBayes(model, sentence);//testing phse

    for (size_t i = 0; i < sentence.size(); ++i) {//output
        cout << sentence[i] << "/" << tags[i] << " ";
    }
    cout << endl;

    return 0;
}
