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
import matplotlib.pyplot as plt
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn import preprocessing

# define functions
def getData(fname):
    df = pd.read_csv(fname)
    # get texts & labels
    texts = df["description"] 
    print("Texts: "+str(type(texts[0])))
    trueLabs = df["sentiment"]
    print("Labels: "+str(type(trueLabs)))
    return texts, trueLabs

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
        total += float(num)**2 # Here
    return total**(1/2)

def normalizeVec(vecIn:list):
    """This function normalizes a vector to unit length.
        :param vecIn:  a list representing a vector, one element per dimension.
        :return: a list representing a vector, that has been normalized to unit length.
    """
    normVec = []
    for num in vecIn:
        if getVecLength(vecIn) > 0: # add if-else to deal with vectors of length 0
            normVec.append(float(num)/getVecLength(vecIn)) #HERE ZeroDivisionError: float division by zero
        else:
            normVec.append(0)
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
        cosAB = dotProductVec(normalizeVec(vecInA), normalizeVec(vecInB)) # here
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
    n = len(vecDict["hi"]) # number of values in each vector
    nv = len(values) # number of vectors 
    meanVec = [0]*n
    for i in range(n):
        sumVec = 0 
        for v in range(nv): 
            sumVec += float(values[v][i]) 
            sumVec = sumVec/nv
            meanVec[i] = sumVec
    return meanVec

def selectGloVecs(gloveVectors: dict):
    words = ["yes", "positive", "good", "agree", "like", 
            "maybe", "neutral", "okay", "alright", "middle",
            "no", "negative", "bad", "disagree", "dislike"]
    myGloVes = {}
    for w in words:
        if w in gloveVectors.keys():
            myGloVes[w] = gloveVectors[w]
    return myGloVes

def getCSDicts(texts:list, gloves:dict, selected:dict):
    csDicts = []
    for txt in texts:
        toks = tokenizeDoc(txt) 
        centroid = computeCentroidVector(toks, gloves)
        cosSim = {}
        # get cosine similarity to some +/- gloVecs
        for w in selected.keys(): 
            cs = cosine(centroid, selected[w]) # doesn't like this line
            cosSim[w] = cs
        csDicts.append(cosSim)
    return csDicts

def scoresToArray(selected:dict, csDictList:list, featureArray):
    # for each text, add each cosine similarity value from csDicts to vecArray 
    allScores = []                  
    for k in selected.keys():
        scores = []     
        for d in csDictList:
            scores.append(d[k]) # HERE TypeError: only integer scalar arrays can be converted to a scalar index when converting d[k] to list
        allScores.append(scores)
    newArray = []
    for i in range(len(allScores)):
        allFeatures = list(featureArray[i]).append(list(allScores[i])) #IndexError: list index out of range
        newArray.append(allFeatures)
    return newArray 

def getSentimentScores(csDict, keys):
    score = 0
    for k in keys:
        score += csDict[k] # divide by number of keys?
    return score

#def assignLabels():

def main():

    # read in train, validation or test set
    #trainDf "data/train_instactivism_60.csv" X
    #validateDf "data/validate_instactivism_20.csv"
    #testDf "data/test_instactivism_20.csv"
    dataSubset = "data/train_instactivism_60.csv" #choose train, validate or test
    # get texts & labels
    texts, trueLabs = getData(dataSubset)

    # get pre-trained GloVe embeddings
    fname = "data/glove.twitter.27B.25d.txt"
    gloVecs = loadGloveVectors(fname) # takes a bit to load

    # select a few positive,  neutral and negative word embeddings to compare to centroids ?
    myGloVes = selectGloVecs(gloVecs)
    posKeys = ["yes", "positive", "good", "agree", "like"]
    neuKeys = ["maybe", "neutral", "okay", "alright", "middle"]
    negKeys = ["no", "negative", "bad", "disagree", "dislike"]

    # get dictionary of centroid vectors of tokenized descriptions for each text
    csDicts = getCSDicts(texts, gloVecs, myGloVes) # list of dicts
    print(len(csDicts))
    print(csDicts[0])

    # get unigram feature matrix 
    cv = CountVectorizer(ngram_range=(1,1)) # not much changes when bigrams are added, stop_words="english" improve pos, make neu/neg acuracy worse
    cleanTexts = cleanAllTexts(texts)
    cvFit = cv.fit(cleanTexts)
    vecArray = cv.transform(cleanTexts).toarray()

    # for each text, add each cosine similarity value from csDicts to vecArray 
    newArray = scoresToArray(myGloVes, csDicts, vecArray)
    print(len(newArray))
    print(len(newArray[1]))

    """     fancyArray = []
    for i in vecArray:
        allFeatures = np.append(vecArray[i], list(allScores[i]))
        fancyArray.append(allFeatures) """



"""         scoreDict = {}
        scoreDict["positive"] = getSentimentScores(cosSim, posKeys)
        scoreDict["neutral"] = getSentimentScores(cosSim, neuKeys)
        scoreDict["negative"] = getSentimentScores(cosSim, negKeys)
        predLabs.append(max(scoreDict, key = scoreDict.get)) # predict label with highest score """
        
"""     print(cosSim)
    print(scoreDict)
    print(len(predLabs))
    #predLabs = predLabs

    cm = confusion_matrix(trainLabs, predLabs, labels=["negative", "neutral", "positive"])
    cmd = ConfusionMatrixDisplay(cm, display_labels=["negative", "neutral", "positive"])
    cmd.plot()
    plt.show() """

"""     # get accuracy, f1, precision and recall
    accuracy = accuracy_score(trainLabs, predLabs) 
    print("Accuracy: " + str(accuracy))
    f1 = f1_score(trainLabs, predLabs, average = "macro") 
    print("F1: " + str(f1))
    precision = precision_score(trainLabs, predLabs, average = "macro") 
    print("Precision: " + str(precision))
    recall = recall_score(trainLabs, predLabs, average = "macro")
    print("Recall: " + str(recall))
    report = classification_report(trainLabs, predLabs)
    print(report) """

    #print(len(centroids))
    #print(centroids[0])
if __name__== "__main__" :
    main()