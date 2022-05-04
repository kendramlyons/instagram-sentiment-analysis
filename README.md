# instagram-sentiment-analysis
Final project for INFO 539: Statistical Natural Lanaguage Processing

Pre-trained GloVe embeddings from: https://nlp.stanford.edu/projects/glove/

Emotional appeals and opinions are often expressed in the captions of Instagram and other social media posts made by activists, grassroots organizations and non-profit businesses. Examining these posts with sentiment analysis can be useful for understanding how positive, neutral or negative they are. Pre-trained GloVe embeddings are used for measuring semantic similarity and dissimilarity of words and texts. This project used a supervised machine learning algorithm to classify text from Instagram posts based on cosine similarity between the centroid vectors of the text and selected GloVe vectors of positive, neutral and negative terms. A logistic regression classifier was used to predict the sentiment of Instagram posts as positive, neutral or negative. This approach yielded a small performance improvement compared to a baseline bag-of-words (BoW) logistic regression model. 
