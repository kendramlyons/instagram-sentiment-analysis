# Non-parametric bootstrap resampling
import pandas as pd
import random
from sklearn import preprocessing

def oneResample(differenceScores: list):
    len_rndm_smpl = len(differenceScores)
    bsample = []
    for i in range(0, len_rndm_smpl):
        sample_idx = random.randrange(0, len_rndm_smpl)
        bsample.append(differenceScores[sample_idx])
    return bsample

def calculatePValue(differenceScores: list, numResamples = 10000):
    h = 0
    j = 0
    for i in range(0,numResamples):
        resample = oneResample(differenceScores)
        sum_resample = sum(resample)
        if (sum_resample > 0):
            h += 1
        else:
            j += 1
        #get p value
    p_value = float(j)/float(numResamples)
    return p_value

def main():
    random.seed(13)
    allResults = pd.read_csv("data/validate_predictions_instactivism.csv") #"data/test_predictions_instactivism.csv"

    # encode scores and add to dataframe    
    le = preprocessing.LabelEncoder()

    encodedBline = le.fit_transform(allResults['baseline'])
    allResults['encodeBline'] = encodedBline

    encodedExp = le.transform(allResults['experimental'])
    allResults['encodeExp'] = encodedExp

    differenceScores = encodedExp-encodedBline
    allResults['differenceScores'] = differenceScores

    pValue = calculatePValue(differenceScores)
    print("P-value: "+str(pValue))

    allResults.to_csv("data/validate_differences_instactivism.csv")

if __name__== "__main__" :
    main()