# ERROR ANALYSIS #
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # get data
    train = pd.read_csv("data/train_50D_errors.csv")
    validate = pd.read_csv("data/validate_50D_errors.csv")
    test = pd.read_csv("data/test_50D_errors.csv")
    
    # combine dataframes into one
    allErrors = [train, validate, test]
    allErrors = pd.concat(allErrors)
    allErrors["error type"] = allErrors["sentiment"] +"-"+ allErrors["predicted"]
    
    # get error-types and counts
    errorTypes = allErrors["error type"].unique() # all possible error types
    errorCounts = allErrors["error type"].value_counts() # pandas series
    # make dataframe 
    errorDf = pd.DataFrame(errorCounts)
    errorDf["error_type"] = errorDf.index
    errorDf.index = range(1, 7) 
    errorDf.columns = ["n", "error_type"]
    # add percent column
    errorDf["percent"] = errorDf["n"]/sum(errorDf["n"])

    print("Positive misclassifications: " + str(round((errorDf.percent[1]+errorDf.percent[2])*100, 1))+"%")
    print("Neutral misclassifications: " + str(round((errorDf.percent[3]+errorDf.percent[4])*100, 1))+"%")
    print("Negative misclassifications: " + str(round((errorDf.percent[5]+errorDf.percent[6])*100, 1))+"%")

    # plot error counts by type
    plt.figure()
    plt.bar(errorDf.error_type, errorDf.percent, color = "darkgreen")
    plt.xlabel("Error Types (true-predicted)")
    plt.title("Error Type Proportions")
    plt.show()


if __name__== "__main__" :
    main()