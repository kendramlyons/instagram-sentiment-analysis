'''
Author: Kendra Lyons
Date: 04/17/2022

This script performs a baseline unigram-based sentiment analysis task on text from Instagram posts 
that use any of the following hashtags: #grassroots, #communityorganization/s, #filibuster, #protest/s,
#mutualaid, #socialmovement/s. The data were labeled by hand without taking emoji into account.

It defines functions for cleaning the texts, trains a logistic regression classifier on 60% of the 
data set and uses it to make predictions of positive, neutral or negative sentiment on a validation 
or test set (each containing 20% of the original data). It plots a confusion matrix and prints a 
performance report with accuracy, F1, precision and recall scores by class and overall.

'''

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd


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


def main():

    # read in data
    trainDf = pd.read_csv("data/train_instactivism_60.csv")
    validateDf = pd.read_csv("data/validate_instactivism_20.csv")
    testDf = pd.read_csv("data/test_instactivism_20.csv")

    # get training docs
    trainTexts = trainDf["description"]
    # get unigram feature matrix 
    cv = CountVectorizer(ngram_range=(1,1)) # not much changes when bigrams are added, stop_words="english" improve pos, make neu/neg acuracy worse
    cleanTexts = cleanAllTexts(trainTexts)
    model = cv.fit(cleanTexts)
    vecArray = cv.transform(cleanTexts).toarray() # keep toarray (fancy)
    #features = cv.vocabulary_

    # get labels
    trainLabels = trainDf["sentiment"]
    le = preprocessing.LabelEncoder()
    encodedLabs = le.fit_transform(trainLabels)

    # train logistic regression classifier
    lrc = LogisticRegression(class_weight = "balanced") 
    lrModel = lrc.fit(vecArray, encodedLabs)

    # set validation or test 
    test_df = validateDf #testDf
    # test set data
    testText = cleanAllTexts(test_df["description"])
    testLabels = test_df["sentiment"]
    testArray = cv.transform(testText).toarray()
    # predict 
    numPreds = lrModel.predict(testArray)
    testPreds = le.inverse_transform(numPreds)

    # inspect errors
    cm = confusion_matrix(testLabels, testPreds, labels=["negative", "neutral", "positive"]) #, normalize = "true"
    cmd = ConfusionMatrixDisplay(cm, display_labels=["negative", "neutral", "positive"])
    cmd.plot()
    plt.show()

    # get accuracy, f1, precision and recall
    accuracy = accuracy_score(testLabels, testPreds) 
    print("Accuracy: " + str(accuracy))
    f1 = f1_score(testLabels, testPreds, average = "macro") 
    print("F1: " + str(f1))
    precision = precision_score(testLabels, testPreds, average = "macro") 
    print("Precision: " + str(precision))
    recall = recall_score(testLabels, testPreds, average = "macro")
    print("Recall: " + str(recall))
    report = classification_report(testLabels, testPreds)
    print(report)

    results = pd.DataFrame({"texts": testText, "my_labels":testLabels.values, "predicted": testPreds})
    results.to_csv("data/baseline_test_results_instactivism.csv")

if __name__== "__main__" :
    main()
