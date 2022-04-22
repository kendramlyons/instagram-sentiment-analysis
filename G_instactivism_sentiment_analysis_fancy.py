'''
Author: Kendra Lyons
Date: 4/21/2022

This script performs an enhanced sentiment analysis task using pre-trained GloVe embeddings on text 
from Instagram posts that use any of the following hashtags: #grassroots, #communityorganization/s, 
#filibuster, #protest/s, #mutualaid, #socialmovement/s. The data were labeled by hand without taking 
emoji into account.

It defines functions for cleaning the texts, trains a logistic regression classifier on 60% of the 
data set and uses it to make predictions of positive, neutral or negative sentiment on a validation 
or test set (each containing 20% of the original data). It plots a confusion matrix and prints a 
performance report with accuracy, F1, precision and recall scores by class and overall.

pre-trained GloVe embeddings from: https://nlp.stanford.edu/projects/glove/
'''
# load packages
#import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn import preprocessing

# define functions
def cleanDoc(oneDoc: str): 
    characters = "<>#_/-.,:;@"
    for c in characters:
        oneDoc = oneDoc.replace(c, " ")
    cleaned = oneDoc.replace("!", " !").replace("\n", " ").replace("   ", " ").replace("  ", " ").strip()
    return cleaned

def cleanAllTexts(allDocs: list):
    cleanTexts = []
    for i in range(len(allDocs)):
        clean = cleanDoc(allDocs[i])
        cleanTexts.append(clean) 
    return cleanTexts

def tokenizeDoc(oneDoc: str): # cleans AND tokenizes
    tokens = cleanDoc(oneDoc).lower().split(" ")
    for t in tokens:
        if len(t)<2 and t!="!":
            tokens.remove(t)
    return tokens

def tokenizeAll(allDocs:list):
    tokenized = []
    for i in range(len(allDocs)):
        tokens = tokenizeDoc(allDocs[i])
        tokenized.append(tokens)
    return tokenized

def getVecLength(vecIn: list):
    """This function computes the length of a vector.
        :param vecIn: a list representing a vector, one element per dimension.
        :return: the length of the vector.
    """
    total = 0
    for num in vecIn:
        total += num**2
    return total**(1/2)

def normalizeVec(vecIn:list):
    """This function normalizes a vector to unit length.
        :param vecIn:  a list representing a vector, one element per dimension.
        :return: a list representing a vector, that has been normalized to unit length.
    """
    normVec = []
    for num in vecIn:
        normVec.append(num/getVecLength(vecIn))
    return normVec

def dotProductVec(vecInA:list, vecInB:list):
        """This method takes the dot product of two vectors.
        :param vecInA, vecInB: two lists representing vectors,
            one element per dimension.
        :return: the dot product.
        """
        dp = 0
        for i in range(len(vecInA)):
            dp += vecInA[i]*vecInB[i]
        return dp

def cosine(vecInA: list, vecInB: list):
        """This method obtains the cosine between two vectors
            (which is nominally the dot product of two vectors of unit length).
        :param vecInA, vecInB: two lists representing vectors, one element per dimension.
        :return: the cosine.
        """
        cosAB = dotProductVec(normalizeVec(vecInA), normalizeVec(vecInB))
        return cosAB

def loadGloveVectors(fname):
    gloVecs = {}
    with open(fname, 'r', encoding="utf-8") as glove:
        for line in glove:
            values = line.strip().split(" ")
            word = values[0]
            vectors = values[1:] #removed array
            gloVecs[word] = vectors
    glove.close()
    print("Loaded "+str(len(gloVecs.keys()))+" vectors!")
    return gloVecs

def computeCentroidVector(tokensIn:list, vecDict:dict):
    """This method calculates the centroid vector from a list of
        tokens. The centroid vector is the "average"
        vector of a list of tokens.
    #NOTE:  Special considerations:
            - all tokens should be converted to lower case.
            - if a vector isn't in the dictionary, it
                shouldn't be a part of the average.
        :param tokensIn: a list of tokens.
        :param vecDict: the vector library is a dictionary, 'vecDict',
            whose keys are tokens, and values are lists representing vectors.
        :return: the centroid vector, represented as a list.
    """
    vectors = {}
    for tok in tokensIn: # if a token is in glove vectors, add key & values to new dict
        if tok in vecDict.keys():
            vectors[tok] = vecDict[tok]
    values = list(vectors.values()) # get list of glove vectos 
    n = 25 # number of values in each vector
    nv = len(values) # number of vectors 
    meanVec = [0]*n
    for i in range(n):
        sumVec = 0 #moved up
        for v in range(nv): # HERE
            sumVec += float(values[v][i]) # HERE
            sumVec = sumVec/nv
            meanVec[i] = sumVec
    return meanVec

def selectGloVecs(gloveVectors: dict):
    words = ["positive", "good", "approve", "like", 
            "neutral", "okay", "alright", "middle",
             "negative", "bad", "disapprove", "dislike"]
    myGloVes = {}
    for w in words:
        if w in gloveVectors.keys():
            myGloVes[w] = gloveVectors[w]
    return myGloVes


def main():

    # read in train, validation and test sets
    trainDf = pd.read_csv("data/train_instactivism_60.csv")
    validateDf = pd.read_csv("data/validate_instactivism_20.csv")
    testDf = pd.read_csv("data/test_instactivism_20.csv")

    # get texts & labels
    trainTexts = trainDf["description"] #changed
    print("Texts: "+str(type(trainTexts[0])))
    trainLabs = trainDf["sentiment"]
    print("Labels: "+str(type(trainLabs)))

    # get pre-trained GloVe embeddings
    fname = "data/glove.twitter.27B.25d.txt"
    gloVecs = loadGloveVectors(fname) # takes a bit to load

    # select a few positive,  neutral and negative word embeddings to compare to centroids ?
    myGloVes = selectGloVecs(gloVecs)

    # get centroid vectors of tokenized descriptions
    tokTexts = []
    centroids = []
    for txt in trainTexts:
        toks = tokenizeDoc(txt) #changed
        centroid = computeCentroidVector(toks, gloVecs) # 
        centroids.append(centroid)
        tokTexts.append(toks)

        # get cosine similarity to some +/- gloVecs

    print(len(centroids))

if __name__== "__main__" :
    main()