'''
Author: Kendra Lyons
Date: 4/9/21

Description:
This program takes a hashtag, gets 50 recent posts that used it
and saves them in a .json file.  

via the Instagram Data API by logicbuilder on RapidAPI.com 
https://rapidapi.com/logicbuilder/api/instagram-data1
'''

import requests
import json

def hashtag_api(hashtag): 
    '''
    takes a string representing hashtag (include quotation marks!), 
    requests and returns a response with data for 50 instagram posts
    from the Instagram Data API by logicbuilder on RapidAPI.com 
    '''

    url = "https://instagram-data1.p.rapidapi.com/hashtag/feed"

    querystring = {"hashtag": hashtag}

    headers = {
        'x-rapidapi-key': "d76115f05amsh3121ad506215084p15c385jsn4a27fc915ff3",
        'x-rapidapi-host': "instagram-data1.p.rapidapi.com"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)
    
    return response


def clean_response(response):
    '''
    takes the response from hashtag_api, extracts and returns the list of dictionaries 
    containing post data
    '''
    
    metad = response.json()
    
    posts = metad["collector"]

    return posts


def save_response(posts, hashtag):
    '''
    saves a json file with the post data
    '''
    fname = hashtag.strip('"')
    fname = fname+'.json'
    
    with open(fname, "w") as f:
        json.dump(posts, f)

def open_saved(fname):
    '''
    opens a saved json file and puts the information into a list of dictionaries
    '''
    file = open("mutualaid.json", "r")
    data = json.load(file)
    return(data)

#==========================================================
def main():
    '''
    Gets a sample of 50 posts containing the specified tag. 
    Prints the type (dict) and content of the first post.
    '''
    tag = "socialmovements" #change tag to try new search term 
    protest = hashtag_api(tag)
    protest

    protest_posts = clean_response(protest)
    print("Type:"+str(type(protest_posts[0]))+"First Post:"+str(protest_posts[0]))
    save_response(protest_posts, tag)

if __name__ == '__main__':
    main()
