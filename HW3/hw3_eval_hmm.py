########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Evaluate the output of your bigram HMM POS tagger
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
import numpy as np

# Class that stores a word and tag together
# Copy from hmm file
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_')
        self.word = parts[0]
        self.tag = parts[1]

# A class for evaluating POS-tagged data
class Eval:
    ################################
    #intput:                       #
    #    goldFile: string          #
    #    testFile: string          #
    #output: None                  #
    ################################
    # Copy from hmm file
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r")  # open the input file in read-only mode
            sens = []
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence)  # append this list as an element to the list of sentences
            return sens
        else:
            print(
                "Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit()  # exit the script

    def __init__(self, goldFile, testFile):
        print("Your task is to implement an evaluation program for POS tagging")
        self.goldData = self.readLabeledData(goldFile)
        self.testData = self.readLabeledData(testFile)
        self.goldTag = defaultdict(float)
        self.testTag = defaultdict(float)
        for i in range(len(self.testData)):
            for j in range(len(self.testData[i])):
                self.testTag[self.testData[i][j].tag] += 1.0

        for i in range(len(self.goldData)):
            for j in range(len(self.goldData[i])):
                self.goldTag[self.goldData[i][j].tag] += 1.0
        self.confusionMatrix = np.zeros((len(self.goldTag), len(self.goldTag)), dtype = int)

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getTokenAccuracy(self):
        print("Return the percentage of correctly-labeled tokens")
        correct = 0.0
        count = 0.0
        for i in range(len(self.testData)):
            for j in range(len(self.testData[i])):
                if self.testData[i][j].tag == self.goldData[i][j].tag:
                    correct += 1.0
                count += 1.0
        acc = correct / count
        return acc
        # return 1.0

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getSentenceAccuracy(self):
        print("Return the percentage of sentences where every word is correctly labeled")
        count = 0.0
        total = len(self.goldData)
        for i in range(len(self.goldData)):
            correct = 0
            for j in range(len(self.goldData[i])):
                if self.goldData[i][j].tag == self.testData[i][j].tag:
                    correct += 1.0
                if correct == len(self.goldData[i]):
                    count += 1
        acc = count / total
        return acc
        # return 1.0

    ################################
    #intput:                       #
    #    outFile: string           #
    #output: None                  #
    ################################
    def writeConfusionMatrix(self, outFile):
        print("Write a confusion matrix to outFile; elements in the matrix can be frequencies (you don't need to normalize)")
        counts = defaultdict(int)
        ### keys of gold tags
        keys = self.goldTag.keys()
        keys = list(keys)

        for i in range(len(self.goldData)):
            for j in range(len(self.goldData[i])):
                counts[self.goldData[i][j].tag, self.testData[i][j].tag] += 1

        for i in range(len(keys)):
            for j in range(len(keys)):
                self.confusionMatrix[i][j] = counts[keys[i], keys[j]]

        f = open(outFile, 'w+')
        f.write('   '.join(keys) + '\n')
        for i in range(len(self.confusionMatrix)):
            line = self.confusionMatrix[i].astype(str)
            f.write(keys[i] + '    ' + '   '.join(line) + '\n')
        f.close()
        return self.confusionMatrix

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    def getPrecision(self, tagTi):
        print("Return the tagger's precision when predicting tag t_i")
        gold_count = 0.0
        test_count = 0.0
        for i in range(len(self.goldData)):
            for j in range(len(self.goldData[i])):
                if self.testData[i][j].tag == tagTi:
                    test_count += 1.0
                    if self.goldData[i][j].tag == tagTi:
                        gold_count += 1.0
        return gold_count / test_count
        # return 1.0

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    # Return the tagger's recall on gold tag t_j
    def getRecall(self, tagTj):
        print("Return the tagger's recall for correctly predicting gold tag t_j")
        gold_count = 0.0
        test_count = 0.0
        for i in range(len(self.goldData)):
            for j in range(len(self.goldData[i])):
                if self.goldData[i][j].tag == tagTj:
                    gold_count += 1.0
                    if self.testData[i][j].tag == tagTj:
                        test_count += 1.0
        return test_count / gold_count
        # return 1.0


if __name__ == "__main__":
    # Pass in the gold and test POS-tagged data as arguments
    if len(sys.argv) < 2:
        print("Call hw2_eval_hmm.py with two arguments: gold.txt and out.txt")
    else:
        gold = sys.argv[1]
        test = sys.argv[2]
        # You need to implement the evaluation class
        eval = Eval(gold, test)
        # Calculate accuracy (sentence and token level)
        print("Token accuracy: ", eval.getTokenAccuracy())
        print("Sentence accuracy: ", eval.getSentenceAccuracy())
        # Calculate recall and precision
        print("Recall on tag NNP: ", eval.getPrecision('NNP'))
        print("Precision for tag NNP: ", eval.getRecall('NNP'))
        # Write a confusion matrix
        eval.writeConfusionMatrix("confusion_matrix.txt")
