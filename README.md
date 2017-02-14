# Microblogging-Clustering-ML
We present a text representation framework by harnessing the power of semantic knowledge bases, i.e., Wikipedia and Wordnet. The originally uncorrelated texts are connected with the semantic representation, thus it enhances the performance of short text clustering and labelling. The experimental results on Twitter dataset demonstrate the superior performance of our framework in handling noisy and short micro-blogging messages.
The feature space is processed using unsupervised machine learning techniques. In this we try to find hidden structures from unlabelled data and then use K-Means clustering technique, a popular method for cluster analysis in data mining. The resultant clusters are labelled according to the highest informative score of the word contained in the tweets of that cluster using RAKE algorithm.

## Dataset
To extract the dataset, Twitter’s Search API, which is a part of Twitter’s REST API is used. It works just like the search feature of Twitter and searches against recent tweets published in the past 7 days. Search API is based on relevance and not completeness. This means that not all users and tweets may be present in the search result. For completeness and realtime retrieval, Streaming API is preferred. 

## Project Flow
*Syntactic Decomposition
*Semantic Mapping using knowledge bases like Wordnet and Wikipedia.
*Clustering using K Means
*Labelling using RAKE

## Contribution
Feel free to contribute and suggest new techniques to make this project better.

## Project Contributors
Rishabh Gupta
Sachin Agarwal
Shreynik Kumar
Vanshaj Behl

**We hope, this project will help you give a start to machine learning world. **
