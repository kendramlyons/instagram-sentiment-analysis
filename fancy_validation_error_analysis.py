# ERROR ANALYSIS #
from ftplib import all_errors
import pandas as pd
import matplotlib.pyplot as plt

valResults = pd.read_csv("data/results_validate_instactivism_20.csv", index_col=0)
valResults["evaluation"] = "validate"
testResults = pd.read_csv("data/results_test_instactivism_20.csv", index_col=0)
testResults["evaluation"] = "test"

validation_errors = valResults[valResults.my_labels != valResults.predicted] 
print(str(len(validation_errors))+" validation errors")

test_errors = testResults[testResults.my_labels != testResults.predicted]
print(str(len(test_errors))+" test errors")

all_errors = validation_errors.append(test_errors)
print(str(len(all_errors))+" total errors")

all_errors.to_csv("data/all_errors_fancy.csv")