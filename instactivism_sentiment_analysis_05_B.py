'''
Author: Kendra Lyons
Date: 04/17/2022

This script defines functions for cleaning text scraped from Instagram, 
as well as functions that perform a baseline unigram-based sentiment analysis 
classification task with logistic regression.

THERE IS A BUG IN THIS SCRIPT, LINE 81 THAT THROWS AN ERROR: 
raise ValueError("y contains previously unseen labels: %s" % str(diff))
ValueError: y contains previously unseen labels: ['negative' 'neutral' 'positive']
'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd


def cleanDoc(oneDoc: str): # JUST cleans
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

def vectorizeTexts(train_df: pd.DataFrame()): # cleans and vectorizes
    # get list of texts
    trainTexts = train_df["description"]
    # get unigram feature matrix 
    cleanTexts = cleanAllTexts(trainTexts)
    cv.fit(cleanTexts)
    vecArray = cv.transform(cleanTexts).toarray()
    return vecArray

def encodeLabels(train_df: pd.DataFrame):
    #get training labels and encode 
    trainLabs = train_df["sentiment"]
    encodedLabs = le.fit_transform(trainLabs)
    return encodedLabs

def getPredictedLabels(lrcModel, testArray):
    # predict labels
    testPreds = lrcModel.predict(testArray) # HERE
    return testPreds

def checkPerformance(testLabs, testPreds):
    # convert numbers back to text labels
    testPreds = le.inverse_transform(testPreds)
    # inspect validation set errors
    cm = confusion_matrix(testLabs, testPreds, labels=["negative", "neutral", "positive"], normalize = "true") #
    return cm

# read in data
trainDf = pd.read_csv("data/train_instactivism_60.csv")
validateDf = pd.read_csv("data/validate_instactivism_20.csv")
testDf = pd.read_csv("data/test_instactivism_20.csv")

# call count vectorizer and label encoder
cv = CountVectorizer(ngram_range=(1,1)) #, max_df = .8
le = preprocessing.LabelEncoder()

# get training texts & labels
trainArray = vectorizeTexts(trainDf)
encodedLabs = encodeLabels(trainDf)

# fit logistic regression model
lrc = LogisticRegression(class_weight = "balanced") 
lrcModel = lrc.fit(trainArray, encodedLabs)

# get test data and make predictions
testArray = cv.transform(validateDf["description"]).toarray()# change for Validation
testLabels = validateDf["sentiment"] # change for Validation
testPredictions = lrcModel.predict(testArray) # BUG HERE
testResults = checkPerformance(testLabels, testPredictions)
testLabels = le.inverse_transform(testLabels)
# plot and display confusion matrix
cmd = ConfusionMatrixDisplay(testResults, display_labels=["negative", "neutral", "positive"])
cmd.plot()
plt.show()

# print accuracy, f1, precision, recall and report
accuracy = accuracy_score(testLabels, testPredictions) 
print("Accuracy: " + str(accuracy))
f1 = f1_score(testLabels, testPredictions, average = "macro") 
print("F1: " + str(f1))
precision = precision_score(testLabels, testPredictions, average = "macro") 
print("Precision: " + str(precision))
recall = recall_score(testLabels, testPredictions, average = "macro")
print("Recall: " + str(recall))
report = classification_report(testLabels, testPredictions)
print(report)


