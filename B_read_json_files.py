import json
import pandas as pd
import numpy as np

paths_21 = ["filibuster/fillibuster4_09.json", 
        "4_09_21/communityorganizations.json", 
        "4_09_21/filibuster.json", "4_09_21/grassroots.json", 
        "4_09_21/mutualaid.json", "4_09_21/protest.json", 
        "4_09_21/protests.json", "4_09_21/socialmovements.json",
        "4_08_21/grassroots.json", "4_08_21/socialmovements.json"]

paths_22 = ["4_06_22/communityorganization.json",
        "4_06_22/communityorganizations.json", 
        "4_06_22/filibuster.json", "4_06_22/grassroots.json",
        "4_06_22/protest.json", "4_06_22/protests.json",
        "4_06_22/socialmovement.json", "4_06_22/socialmovements.json"]

with open("filibuster/fillibuster4_09.json") as json_file:
        filibuster_4_09 = json.load(json_file) #loads a single list of dicts

type(filibuster_4_09) # list
type(filibuster_4_09[1]) # dict

data_list = []
for path in paths_21:
    with open(path) as json_file:
        data = json.load(json_file) # each json file a list of dicts
        for dictionary in data: # iterate over dictionaries in list
            data_list.append(dictionary) # a list of dictionaries

data_list[0]["description"]

data_list[0]["hashtags"]

len(data_list)
# 177363 
data_list[402] #dictionary
data_list[403] #character

data_list = data_list[0:402]

# create a list of tuples () #TEST new features
text_data = []
for post in data_list: # FIXED TypeError: string indices must be integers (only first 403 objects are dicts)
    post_id = post["id"] 
    description = post["description"]
    len_description = len(post["description"])
    hashtags = post["hashtags"] #number of hashtags, comments, likes and mentions?
    num_hasht = len(post["hashtags"])
    comments = post["comments"]
    likes = post["likes"]
    mentions = len(post["mentions"])
    text_data.append([post_id, description, len_description, hashtags, 
                        num_hasht, comments, likes, mentions]) # creates a list of len 403 (not all data?)

# create dataframe from list of tuples
ig_df = pd.DataFrame(text_data, columns = ["post_id", "description", "len_description", "hashtags", 
                                            "num_hasht", "comments", "likes", "mentions"])

# find out how many unique posts there are
len(np.unique(ig_df['post_id'])) # 352 unique posts

ig_df = ig_df.drop_duplicates(subset=["post_id", "description"]) #adding duplicates may not do anything here

ig_df.to_csv("insta_activism_text_2021_plus.csv", index=False)

