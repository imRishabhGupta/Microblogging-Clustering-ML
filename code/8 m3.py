# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 22:17:29 2016

@author: lenovo
"""
from __future__ import division
from nltk import Tree
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk.corpus import stopwords
import rake
import nltk
import numpy as nm
import pickle 
import pandas as pd
from nltk.corpus import wordnet as wn
import re
from sklearn.cluster import KMeans
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer

    
stemmer = SnowballStemmer("english")
print "started"
def rem_reduncy(seq): # Dave Kirby
    # Order preserving
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]
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


with open('tweets.json', 'rb') as f:
 tweets = pickle.load(f)
with open('labels.json', 'rb') as f:
 labels = pickle.load(f)
features_word=[]
features_phrase=[]
text=[]
ts=time.time()
for j in range(len(tweets)):
    tokens = nltk.word_tokenize(tweets[j])
    tagged = nltk.pos_tag(tokens)
    #print tagged
    for i in range(len(tagged)):
        if tagged[i][1]=='VBG' or tagged[i][1]=='VBD' or tagged[i][1]=='VBZ' or tagged[i][1]=='VBP' or tagged[i][1]=='CC' or tagged[i][1]=='WP' or tagged[i][1]=='NNP' or tagged[i][1]=='JJ' or tagged[i][1]=='NN' or tagged[i][1]=='NNS':
            features_word.append(tagged[i][0])
    text.append(Tree('S',tagged))
for word in features_word: # iterate over word_list
  if word in stopwords.words('english'): 
    features_word.remove(word)
features_word = [stemmer.stem(t) for t in features_word]

semantic_features_word=[]  
for word in features_word : 
    for i in range(5):
        try:    
            for j in wn.synset(word+'.n.'+str(i)).lemma_names():
                semantic_features_word.append(j)
        except:
            pass
#print len(semantic_features_word)
semantic_features_word = rem_reduncy(semantic_features_word)
#print len(semantic_features_word)

parser = RegexpParser("""
NP: {<DT>? <JJ>* <NN.*>*}    #NP
P: {<IN>}           # Preposition
V: {<V.*>}          # Verb
PP: {<P> <NP>}      # PP -> P NP
VP: {<V> <NP|PP>*}  # VP -> V (NP|PP)*""")
for i in range(len(text)):
    chunked_text = parser.parse(text[i])
    #print (chunked_text)        #whole chunked text
    #chunked_text.draw()             #represent chunked text in a tree
    for subtree in chunked_text.subtrees():
        if subtree.label() == 'NP' or subtree.label() == 'VP':
            feature1=""
            for leaf in  subtree.leaves():
                feature1=feature1+leaf[0]+" "
                feature1=feature1[0:len(feature1)-1]
                if len(subtree)!=1:
                    #print(subtree)
                    features_phrase.append(feature1)
print len(features_word)
print len(features_phrase)
terms=features_word+features_phrase
terms=rem_reduncy(terms)
print len(terms)

with open('semantics_phrases_wiki.txt', 'r') as f:
   features_wiki=pickle.load(f)

terms=terms + features_wiki + semantic_features_word
terms=rem_reduncy(terms)
tfidf_vectorizer = TfidfVectorizer( max_features=200000, stop_words='english', vocabulary=terms,
                                 use_idf=True, tokenizer=tokenize_and_stem)

tfidf_matrix = tfidf_vectorizer.fit_transform(tweets)
#print tfidf_matrix

num_clusters = 100
km = KMeans(n_clusters=num_clusters,  max_iter=200, n_init=10)
km.fit(tfidf_matrix)
tf=time.time()
te=tf-ts
print ("time complexity: ",te)
clusters = km.labels_    
#print (clusters)
# implementing labeling

for j in range(100):
    count=0
    for i in range(len(tweets)):
        if clusters[i]==j:
            count = count + 1
    print "for ",j,"th cluster tweet is: ",count

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