#import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

""" 
Split instagram post data into train, validate and test sets 
"""

text_df = pd.read_csv("data/insta_activism_sentiment_text_21_22_reduced.csv", encoding='utf-8', header = 0)

train_data, test_data = train_test_split(text_df, train_size=0.60, random_state=4, stratify=text_df['sentiment'])

test_data, validate_data = train_test_split(test_data, train_size =0.50, random_state=4, stratify = test_data['sentiment'])

print("Training Data: " + str(len(train_data)) + " observations")
print("Validation Data: " + str(len(validate_data)) + " observations")
print("Testing Data: "+ str(len(test_data))+" observations")

train_data.to_csv("data/train_instactivism_60.csv", index=False)
validate_data.to_csv("data/validate_instactivism_20.csv", index=False)
test_data.to_csv("data/test_instactivism_20.csv", index=False)