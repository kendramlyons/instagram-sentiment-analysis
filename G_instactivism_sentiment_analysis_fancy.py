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
    '''Gets the texts and sentiment label from a .csv file.
        Returns 1) texts as a pandas series, and 2) labels, also as a pandas series
    '''
    df = pd.read_csv(fname, header=0)
    # get texts & labels
    texts = df["description"] 
    trueLabs = df["sentiment"]
    return texts, trueLabs

def cleanDoc(oneDoc: str): 
    '''Removes punctuation (except for exclaimation points) and whitespace from a text string
        Returns a clean text string
    '''
    characters = "<>#_/-.,:;@"
    for c in characters:
        oneDoc = oneDoc.replace(c, " ")
    cleaned = oneDoc.replace("!", " !").replace("\n", " ").replace("   ", " ").replace("  ", " ").strip()
    return cleaned

def cleanAllTexts(allDocs: list):
    '''Removes punctuation (except !) and whitespace from a list of text strings
        Returns a list of clean text strings
    '''
    cleanTexts = []
    for i in range(len(allDocs)):
        clean = cleanDoc(allDocs[i])
        cleanTexts.append(clean) 
    return cleanTexts

def tokenizeDoc(oneDoc: str): 
    '''Cleans and tokenizes a text string
        Returns a list of tokens
    '''
    tokens = cleanDoc(oneDoc).lower().split(" ")
    for t in tokens:
        if len(t)<2 and t!="!":
            tokens.remove(t)
    return tokens

def tokenizeAll(allDocs:list):
    '''Cleans and tokenizes a list of text strings
        Returns a list of lists of tokens
    '''
    tokenized = []
    for i in range(len(allDocs)):
        tokens = tokenizeDoc(allDocs[i])
        tokenized.append(tokens)
    return tokenized

def getVecLength(vecIn: list):
    """Computes the length of a vector.
        :param vecIn: a list representing a vector, one element per dimension.
        :return: the length of the vector.
    """
    total = 0
    for num in vecIn:
        total += float(num)**2 
    return total**(1/2)

def normalizeVec(vecIn:list):
    """Normalizes a vector to unit length.
        :param vecIn:  a list representing a vector, one element per dimension.
        :return: a list representing a vector, that has been normalized to unit length.
    """
    normVec = []
    for num in vecIn:
        if getVecLength(vecIn) > 0: # add if-else to deal with vectors of length 0
            normVec.append(float(num)/getVecLength(vecIn)) #HERE ZeroDivisionError: float division by zero (fixed)
        else:
            normVec.append(0)
    return normVec

def dotProductVec(vecInA:list, vecInB:list):
        """Takes the dot product of two vectors.
        :param vecInA, vecInB: two lists representing vectors,
            one element per dimension.
        :return: the dot product.
        """
        dp = 0
        for i in range(len(vecInA)):
            dp += vecInA[i]*vecInB[i]
        return dp

def cosine(vecInA: list, vecInB: list):
        """Obtains the cosine between two vectors.
        :param vecInA, vecInB: two lists representing vectors, one element per dimension.
        :return: the cosine.
        """
        cosAB = dotProductVec(normalizeVec(vecInA), normalizeVec(vecInB)) 
        return cosAB

def loadGloveVectors(fname):
    '''Loads vectors from a .txt file of pre-trained GloVe embeddings
        Returns a dictionary 
    '''
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
    """Calculates the centroid vector from a list of tokens. 
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
    '''Selects positive, neutral and negative keys & values from a dictionary of 
    pre-trained GloVe vectors to use for estimating sentiment of each word in a texts.
        Returns a dictionary with a subset of positive, neutral, and negative word vectors. 
    '''
    words = ["yes", "positive", "good", "agree", "like", "happy", #"thank",
            "maybe", "neutral", "okay", "alright", "middle", "might", "unsure", "moderate", "inform",
            "no", "negative", "bad", "disagree", "dislike", "not", "hate", "despise", "angry", "mad", "pathetic", "ugh", "stupid"] # add "don't"
    myGloVes = {}
    for w in words:
        if w in gloveVectors.keys():
            myGloVes[w] = gloveVectors[w]
    return myGloVes

def getCSDicts(texts:list, gloves:dict, selected:dict):
    '''Gets a list of one dictionary per text. Each dictionary contains the cosine similarity of 
    the centroid vector of each text and a selected positive, neutral or negative GloVe vector.
    Returns a list of dictionaries with cosine similarity to selected GloVe vectors, one for each text
    '''
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

def scoresToArray(csDictList:list): #selected:dict, 
    """ Takes a dictionary with positive neutral and negative cosine similarity scores 
            Returns a list of lists with the scores, one for each text"""
    allScores = []
    for d in csDictList:
        allScores.append(list(d.values()))
    return allScores 


def main():
    # get pre-trained GloVe embeddings
    fname = "data/glove.twitter.27B.50d.txt" #"data/glove.twitter.27B.25d.txt"
    gloVecs = loadGloveVectors(fname) # takes a bit to load

    # select a few positive, neutral and negative word embeddings to compare to centroids
    myGloVes = selectGloVecs(gloVecs)

    # get training texts & labels
    trainTexts, trainLabs = getData("data/train_instactivism_60.csv")

    # get dictionary of centroid vectors of tokenized descriptions for each text
    csDicts = getCSDicts(trainTexts, gloVecs, myGloVes) # list of dicts TRAINING

    # for each text, add each cosine similarity value from csDicts to vecArray 
    scoreArray = np.array(scoresToArray(csDicts)) # TRAIN 

    # get unigrams and combine arrays
    cv = CountVectorizer(ngram_range=(1,1)) # not much changes when bigrams are added, stop_words="english" improve pos, make neu/neg acuracy worse
    cleanTexts = cleanAllTexts(trainTexts)
    model = cv.fit(cleanTexts)
    vecArrayTrain = cv.transform(cleanTexts).toarray()

    trainArray = []
    for i in range(len(vecArrayTrain)): 
        trainArray.append(np.append(vecArrayTrain[i], scoreArray[i]))

    # encode labels
    le = preprocessing.LabelEncoder()
    encodedLabs = le.fit_transform(trainLabs)

    # train model
    lrc = LogisticRegression(class_weight = "balanced") 
    lrModel = lrc.fit(trainArray, encodedLabs) # TRAIN

    # read in validation or test set and get texts and labels for evaluation  
    #validateDf "data/validate_instactivism_20.csv"
    #testDf "data/test_instactivism_20.csv"
    evalData = "data/test_instactivism_20.csv" # change evaluation texts
    evalTexts, evalLabs = getData(evalData) # EVAL

    csDictsEval = getCSDicts(evalTexts, gloVecs, myGloVes) #EVAL
    scoreArrayEval = scoresToArray(csDictsEval) 

    cleanEvalTexts = cleanAllTexts(evalTexts)
    vecArrayEval = cv.transform(cleanEvalTexts).toarray() 
    print("Vector array length: "+str(len(vecArrayEval)))
    print("Single vector length: "+str(len(vecArrayEval[0])))

    fancyArray = []
    for i in range(len(vecArrayEval)): 
        fancyArray.append(np.append(vecArrayEval[i], scoreArrayEval[i])) 
    print("Fancy array length: "+str(len(fancyArray)))
    print("Single vector length: "+str(len(fancyArray[0])))

    # use model to predict on validate/test texts
    numLabs = lrModel.predict(fancyArray) # EVAL 
    predLabs = le.inverse_transform(numLabs)

    # show confusion matrix
    cm = confusion_matrix(evalLabs, predLabs, labels=["negative", "neutral", "positive"]) #, normalize="true"
    cmd = ConfusionMatrixDisplay(cm, display_labels=["negative", "neutral", "positive"])
    cmd.plot()
    plt.savefig("figures/confusion_num_"+evalData[5:-4]+".png")
    plt.show()

    # print accuracy, f1, precision and recall
    accuracy = accuracy_score(evalLabs, predLabs) # EVAL
    print("Accuracy: " + str(accuracy))
    f1 = f1_score(evalLabs, predLabs, average = "macro") 
    print("F1: " + str(f1))
    precision = precision_score(evalLabs, predLabs, average = "macro") 
    print("Precision: " + str(precision))
    recall = recall_score(evalLabs, predLabs, average = "macro")
    print("Recall: " + str(recall))
    report = classification_report(evalLabs, predLabs)
    print(report)

    results = pd.DataFrame({"texts": evalTexts.values, "my_labels":evalLabs.values, "predicted": predLabs, "prediction_class": evalLabs+"-"+predLabs, "clean_texts": cleanEvalTexts})
    results.to_csv("data/results_"+evalData[5:])

if __name__== "__main__" :
    main()