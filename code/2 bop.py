# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:45:54 2016

@author: lenovo
"""
from __future__ import division
import numpy as nm
import pickle 
import pandas as pd
import nltk
import re
import rake
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans    
stemmer = SnowballStemmer("english")


#defining things

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


#parsing tweets from json file

with open('tweets.json', 'rb') as f:
    tweets = pickle.load(f)

with open('labels.json', 'rb') as f:
    labels = pickle.load(f)


#implementing tfidf
    ts = time.time()
    tfidf_vectorizer = TfidfVectorizer( max_features=200000, stop_words='english',ngram_range=(1, 2) ,use_idf=True, tokenizer=tokenize_and_stem)
    tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)
    print "vectorizing done."

#implementing kmeans
    
    
    num_clusters=100
    km = KMeans(n_clusters=num_clusters,  max_iter=100, n_init=10)
    km.fit(tfidf_matrix)
    print "clustering done"
    clusters = km.labels_ 
    table = { 'tweets': tweets, 'cluster': clusters}
    frame = pd.DataFrame(table, index = [clusters] , columns = ['cluster'])
    tf = time.time()
    te = tf-ts
    print ("Time complexity of clustering: ",te) 


# implementing labeling
    cluster_label = []
    for j in range(100):
        content = ""
        for i in range(len(tweets)):
            if clusters[i]==j:
                content = content + " " + tweets[i]                
        rake_object = rake.Rake("SmartStoplist.txt",4,3,4)             
        keywords = rake_object.run(content)
        score = 0 
        for k in range(len(keywords)):
            if keywords[k][1] > score :
                score = keywords[k][1]
                x = keywords[k][0]
        cluster_label.append(x)


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



