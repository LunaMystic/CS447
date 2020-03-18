########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Train a bigram HMM for POS tagging
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from math import log

# Unknown word token
UNK = 'UNK'

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_')
        self.word = parts[0]
        self.tag = parts[1]

# Class of viterbi that has a backpointer of the path
class Viterbi:
    def __init__(self):
        self.viterbi = float("-inf")
        self.backpointer = ""

# Class definition for a bigram HMM
class HMM:
### Helper file I/O methods ###
    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = []
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = []
            for line in file:
                sentence = line.split() # split the line into a list of words
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s ddoes not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script
### End file I/O methods ###

    ################################
    #intput:                       #
    #    unknownWordThreshold: int #
    #output: None                  #
    ################################
    # Constructor
    def __init__(self, unknownWordThreshold=5):
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        ### Initialize the rest of your data structures here ###
        self.t_tag = defaultdict(float)
        self.tag = defaultdict(float)
        self.word = defaultdict(float)
        self.word_tag = defaultdict(float)
        self.tre = []


    def preprocess(self, data):
        fre = defaultdict(float)
        ### Count occurrance
        for i in range(len(data)):
            for j in range(len(data[i])):
                fre[data[i][j].word] += 1.0
        ### Change the word to unk if its occurance is less than thershold
        for sen in data:
            for i in range(0, len(sen)):
                word = sen[i].word
                if fre[word] < self.minFreq:
                    sen[i].word = UNK

    ################################
    #intput:                       #
    #    trainFile: string         #
    #output: None                  #
    ################################
    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        data = self.readLabeledData(trainFile) # data is a nested list of TaggedWords
        #print("Your first task is to train a bigram HMM tagger from an input file of POS-tagged text")
        self.preprocess(data)
        # print(data[0])
        # print(data[0][0].word)
        # print(data[0][0].tag)
        # exit()
        for i in range(len(data)):
            ### start of the sentence
            self.t_tag[data[i][0].tag, '<s>'] += 1.0
            self.tag[data[i][0].tag] += 1.0
            self.tag['<s>'] += 1.0
            self.word_tag[data[i][0].word, data[i][0].tag] += 1.0
            for j in range(1, len(data[i])):
                self.tag[data[i][j].tag] += 1.0
                self.word[data[i][j].word] += 1.0
                ### Emission
                self.word_tag[data[i][j].word, data[i][j].tag] += 1.0
                ### Transition
                self.t_tag[data[i][j].tag, data[i][j - 1].tag] += 1.0
            ### end of the sentence
            length = len(data[i]) - 1
            self.t_tag['</s>', data[i][length].tag] += 1.0

    def transProb(self, current, previous):
        ### Add-one smoothing of transition probability
        ### since self.tag contains word "<s>", which should not be considered during smoothing
        prob = (self.t_tag[current, previous] + 1) / (self.tag[previous] + len(self.tag) - 1)
        return log(prob)

    def emissionProb(self, word, tag):
        if self.word_tag[word, tag] == 0:
            return float("-inf")
        prob = self.word_tag[word, tag] / self.tag[tag]
        return log(prob)

    def backToSeq(self, length, t, tags):
        i = length
        tagSequence = []
        while (i > 0):
            # print(i)
            tagSequence.insert(0, tags[t])
            t = self.tre[t][i-1].backpointer
            i -= 1
        return tagSequence

    ################################
    #intput:                       #
    #     testFile: string         #
    #    outFile: string           #
    #output: None                  #
    ################################
    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):
        # print("original words")
        data = self.readUnlabeledData(testFile)
        f=open(outFile, 'w+')
        for sen in data:
            vitTags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+vitTags[i]+" "
            # print(vitTags)
            # print(senString)
            print(senString.rstrip(), end="\n", file=f)

    ################################
    #intput:                       #
    #    words: list               #
    #output: list                  #
    ################################
    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        #print("Your second task is to implement the Viterbi algorithm for the HMM tagger")
        # returns the list of Viterbi POS tags (strings)
        self.tre = [[Viterbi() for i in range(len(words))] for j in range(len(self.tag) - 1)]
        tags = self.tag.keys()
        tags = list(tags)
        tags.remove('<s>')
        ### Deal with unknown words without changing the original words in the sentence
        words_copy = list(words)
        for i in range(len(words_copy)):
            if words_copy[i] not in self.word.keys():
                words_copy[i] = UNK
        ### First column of the tre
        for i in range(len(tags)):
            self.tre[i][0].viterbi = self.transProb(tags[i], '<s>') + self.emissionProb(words_copy[0], tags[i])
        for t in range(1, len(words_copy)):
            # print(words_copy[t])
            for j in range(len(tags)):
                self.tre[j][t].viterbi = float("-inf")
                if self.emissionProb(words_copy[t], tags[j]) > float("-inf"):
                    for k in range(len(tags)):
                        if self.tre[k][t - 1].viterbi > float("-inf"):
                            tmp = self.tre[k][t - 1].viterbi + self.transProb(tags[j], tags[k])
                            if tmp > self.tre[j][t].viterbi:
                                self.tre[j][t].viterbi = tmp
                                self.tre[j][t].backpointer = k
                                # print("backpointer", k)
                    self.tre[j][t].viterbi += self.emissionProb(words_copy[t], tags[j])
        t_max = 0
        t = len(words_copy) - 1
        vit_max = float("-inf")
        for j in range(len(tags)):
            if self.tre[j][t].viterbi > float("-inf"):
                value = self.tre[j][t].viterbi
                if value > vit_max:
                    t_max = j
                    vit_max = value
        tagSequence = self.backToSeq(len(words_copy), t_max, tags)
        # print(tagSequence)
        return tagSequence
        # return ["NULL"]*len(words) # this returns a dummy list of "NULL", equal in length to words

if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')
