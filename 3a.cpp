#include <iostream>
#include <string>
#include <cstring>
#include <cctype>

using namespace std;

//word tags definition
string nouns[] = {"cat", "mat", "bat", "rat", "dad", "well"};
string verbs[] = {"sit", "sat", "run", "ran"};
string dets[] = {"a", "an", "the", "this"};
string adverb[]={"quickly","wisely","very"};
string prep[]={"on","at","in","from","is"};

//applying rules for taggers
void getTag(char word[], char tag[]) {
    int len = strlen(word);
    for (const string &noun : nouns) {
        if (strcmp(word, noun.c_str()) == 0) {//if noun
            strcpy(tag, "NN");
            return;
        }
    }
    for (const string &prp : prep) {
        if (strcmp(word, prp.c_str()) == 0) {//if preposition
            strcpy(tag, "PRP");
            return;
        }
    }
    for (const string &verb : verbs) {
        if (strcmp(word, verb.c_str()) == 0) {//if verb
            strcpy(tag, "VB");
            return;
        }
    }
    for (const string &det : dets) {
        if (strcmp(word, det.c_str()) == 0) {//if determiner
            strcpy(tag, "DET");
            return;
        }
    }
    for (const string &det : adverb) {
        if (strcmp(word, det.c_str()) == 0) {//if adverb
            strcpy(tag, "JJ");
            return;
        }
    }
    if (len >= 3 && strcmp(&word[len - 3], "ing") == 0) {//ends with ing
        strcpy(tag, "VBG");
    } else if (len >= 2 && strcmp(&word[len - 2], "ed") == 0) {//ends with ed
        strcpy(tag, "VBD");
    } else if (len >=1 && strcmp(&word[len - 1], "s") == 0) {//ends with s
        strcpy(tag, "NNS");
    } else if (len >= 2 && strcmp(&word[len - 2], "ly") == 0) {//ends with ly
        strcpy(tag, "RB");
    } else if (len >= 4 && strcmp(&word[len - 4], "able") == 0) {//ends with able
        strcpy(tag, "JJ");
    } else   
    strcpy(tag, "NN");//else set as noun
}

void toLowerCase(char *str) {//lower case conversion
    for (int i = 0; str[i]; i++) {
        str[i] = tolower(str[i]);
    }
}

int main() {
    char sentence[] = "the cat is very fat sleeping on the mat";//test sentence
    char taggedSentence[1000] = "";//tagged sentence
    char word[50];
    char tag[10];
    char *token = strtok(sentence, " .");//tokenisation
    while (token != NULL) {
        strcpy(word, token);
        toLowerCase(word);
        getTag(word, tag);//get tag from rules
        char taggedWord[60];
        sprintf(taggedWord, "%s/%s ", word, tag);//assign tag to word in sentence
        if (strlen(taggedSentence) + strlen(taggedWord) < sizeof(taggedSentence)) {
            strcat(taggedSentence, taggedWord);//check for bit errors
        } else {
            cout << "Tagged sentence exceeds buffer size." << endl;
            return 1;  
        }
        token = strtok(NULL, " .");//next token
    }

    cout << "Tagged Sentence:\n" << taggedSentence << endl;

    return 0;
}
