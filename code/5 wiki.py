# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 15:55:20 2016

@author: lenovo
"""
from __future__ import division
from wikiapi import WikiApi
import rake
import numpy as nm
import pickle 
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
import rake
import time
stemmer = SnowballStemmer("english")

    
stemmer = SnowballStemmer("english")

def get_required_tweets(num,tweets,clusters):
    cluster_tweets=[]
    for i in range(len(clusters)):
        if num==clusters[i]:
            cluster_tweets.append(tweets[i])
    return cluster_tweets
    
def get_tp(label_tweets,cluster_tweets):
    count=0
    label_tweets.sort()
    cluster_tweets.sort()
    i=0
    j=0
    while i<len(cluster_tweets) and j<len(label_tweets):
        if cluster_tweets[i]==label_tweets[j]:
            i=i+1
            j=j+1
            count=count+1
        elif cluster_tweets[i]<label_tweets[j]:
            i=i+1
        else :
            j=j+1
    return count

def get_wiki_phrases(word):
    wiki = WikiApi()
    wiki = WikiApi({ 'locale' : 'en'})
    results = wiki.find(word)
    print results
    phrase=""
    for i in range(min(4,len(results))):
        article = wiki.get_article(results[i])
        #print article.content
        phrase=phrase+" "+article.content
        #print phrase
    rake_object = rake.Rake("SmartStoplist.txt",4,3,10) 
    
    #Now, we have a RAKE object that extracts keywords where:
    #   Each word has at least 4 characters
    #   Each phrase has at most 3 words
    #   Each keyword appears in the text at least 4 times
    keywords = rake_object.run(phrase)
    return keywords[0:20]

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems
    
def rem_reduncy(seq): # Dave Kirby
    # Order preserving
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]

with open('tweets.json', 'rb') as f:
 tweets = pickle.load(f)
with open('labels.json', 'rb') as f:
 labels = pickle.load(f)

ts=time.time()
tfidf_vectorizer = TfidfVectorizer( max_features=200000, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem)

tfidf_matrix = tfidf_vectorizer.fit_transform(tweets) #fit the vectorizer to synopses
#print tfidf_matrix
terms=tfidf_vectorizer.get_feature_names()

with open('semantics_words_wiki.txt', 'r') as f:
   features_wiki=pickle.load(f)
#features_wiki=[]
  
features_wiki=rem_reduncy(features_wiki)



features=terms+features_wiki

tfidf_vectorizer = TfidfVectorizer( max_features=200000, stop_words='english', vocabulary=features,
                                 use_idf=True, tokenizer=tokenize_and_stem)

tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)


num_clusters = 100
km = KMeans(n_clusters=num_clusters,  max_iter=200, n_init=10)
km.fit(tfidf_matrix)
tf=time.time()
te=tf-ts
print("time complexity: ",te )
clusters = km.labels_    
#print (clusters)
films = { 'tweets': tweets, 'cluster': clusters}
frame = pd.DataFrame(films, index = [clusters] , columns = ['cluster'])

print ((frame['cluster'].value_counts()))

#evaluating f1score and accuracy

evaluation = nm.zeros(shape=(num_clusters,num_clusters))
precision = nm.zeros(shape=(num_clusters,num_clusters))
recall = nm.zeros(shape=(num_clusters,num_clusters))
label_count = []
acc=0
for i in range(num_clusters):
    label_tweets=get_required_tweets(i,tweets,labels)
    label_count.append(len(label_tweets))
    max_=0
    #print("label no ",i," with count ",label_count[i])
    for j in range(num_clusters):
        cluster_tweets=get_required_tweets(j,tweets,clusters)
        ans=get_tp(label_tweets,cluster_tweets)
        precision[i][j]=float(ans/float(frame['cluster'].value_counts()[j]))
        if ans>max_:
            max_=ans
        if len(label_tweets)==0:
            recall[i][j]=0
        else:
            recall[i][j]=float(ans/(len(label_tweets)))
        evaluation[i][j]=ans
        #print("           cluster no ",j," with count ",len(cluster_tweets)," precision is ",ans) 
    acc=acc+max_

acc=acc/7070
print("accuracy is :",acc)
ans=0
f1=nm.zeros(shape=(num_clusters,num_clusters))
for i in range(num_clusters):
    max_=0
    for j in range(num_clusters):
        f1[i][j]=float((2*precision[i][j]*recall[i][j])/(precision[i][j]+recall[i][j]))
        if f1[i][j]>max_:
            max_=f1[i][j] 
    ans=ans+(label_count[i]*max_)

f1_measure=ans/len(labels)
print("f1 measure is :",f1_measure)
