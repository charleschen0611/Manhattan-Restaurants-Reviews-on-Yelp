#!/usr/bin/env python
# coding: utf-8

# # Project topic: Restaurant Popularity Analysis Using Yelp Reviews<br/>
# - __BIA-660-Group 7__<br/>
# - __Instructor: Prof. Rong Liu__<br/>
# - Group member: Honyi Chen, Tingyi Lu, Junhan Zhou, Xiaomin Yang<br/>
# - 2019 Spring

# In[79]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import pandas as pd
import nltk
#nltk.download()
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt   
from wordcloud import WordCloud
import datetime

import string
import collections

from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
#from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from time import time

#Kmeans Cluster
from nltk.cluster import KMeansClusterer, cosine_distance

import regex as re   

# plot charts inline
get_ipython().run_line_magic('matplotlib', 'inline')


# # Part 1 - Web Scrapping

# - The Web Scraping part is in the other notebook __Web_Scraping_New_V1.ipynb__

# # Part 2 - EDA

# ## import rawData.csv

# In[80]:


df = pd.read_csv('raw_data_0504.csv')
df.head()
df.info() # get detailed information of each column 
df.describe()
print( "\nThere are {} observations in this dataset. \n".format(df.shape[0]))


# In[ ]:





# ## Analysis of rating ditribution of top restaurants

# ## Generate Wordcloud

# In[29]:


# get pie charts of top restaurants

# Peter Luger
df1= df[(df.Restaurant_Name == "Peter Luger")]
df1= df1[['Restaurant_Name','Score']]
df1=df1.groupby('Score').count()
# Pie chart
labels= ['Score=3.0', 'Score=4.0','Score=5.0']
sizes= [df1.iat[0,0]*3.0, df1.iat[1,0]*4.0, df1.iat[2,0]*5.0]
# add colors
colors= ['#ff6666','#99ff99','#66b3ff']
fig1,ax1= plt.subplots()
# explode 1st slice
explode= (0, 0, 0)
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=88)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.title('Ratings of Peter Luger')
plt.tight_layout()



# The Halal Guys
df3= df[(df.Restaurant_Name == "The Halal Guys")]
df3= df3[['Restaurant_Name','Score']]
df3=df3.groupby('Score').count()
# Pie chart
labels= ['Score=3.0', 'Score=4.0','Score=5.0']
sizes= [df3.iat[0,0]*3.0, df3.iat[1,0]*4.0, df3.iat[2,0]*5.0]
# explode 1st slice
explode= (0, 0, 0) 
#add colors
colors= ['#ff6666','#99ff99','#66b3ff']
fig1, ax1= plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=88)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.title('Ratings of The Halal Guys')
plt.tight_layout()

# Carmine's Italian Restaurant - Times Square 
df4= df[(df.Restaurant_Name == "Carmine's Italian Restaurant - Times Square")]
df4= df4[['Restaurant_Name','Score']]
df4=df4.groupby('Score').count()
# Pie chart
labels= ['Score=2.0', 'Score=3.0', 'Score=4.0', 'Score=5.0']
sizes= [df4.iat[0,0]*2.0, df4.iat[1,0]*3.0, df4.iat[2,0]*4.0,df4.iat[3,0]*5.0]
# explode 1st slice
explode= (0, 0, 0, 0) 
#add colors
colors= ['#ff9999','#ff6666','#99ff99','#66b3ff']
fig1, ax1= plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=88)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.title('Ratings of Carmine\'s Italian Restaurant - Times Square')
plt.tight_layout()

#-Ippudo Westside 
# find all scores of The Metropolitan Museum of Art 
df5= df[(df.Restaurant_Name == "Ippudo Westside")]
df5= df5[['Restaurant_Name','Score']]
df5=df5.groupby('Score').count()
# Pie chart
labels= ['Score=3.0', 'Score=4.0', 'Score=5.0']
sizes= [df5.iat[1,0]*3.0, df5.iat[2,0]*4.0, df5.iat[2,0]*5.0]
# explode 1st slice
explode= (0, 0, 0) 
#add colors
colors= ['#ff6666','#99ff99','#66b3ff']
fig1, ax1= plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=88)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.title('Ratings of Ippudo Westside')
plt.tight_layout()



# -Buddakan
df2= df[(df.Restaurant_Name == "Buddakan")]
df2= df2[['Restaurant_Name','Score']]
df2= df2.groupby('Score').count()
# Pie chart
labels= ['Score=3.0', 'Score=4.0', 'Score=5.0']
sizes= [df2.iat[0,0]*3.0, df2.iat[1,0]*4.0, df2.iat[2,0]*5.0]
# explode 1st slice
explode= (0, 0, 0) 
# add colors
colors= ['#ff6666','#99ff99','#66b3ff']
fig1,ax1= plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.title('Ratings of Buddaken')
plt.tight_layout()


# In[31]:


reviews = []
reviews = df.review
#print(reviews[0])
reviews = " ".join([i[ : ] for i in reviews])

#Generate a word cloud image
wordcloud = WordCloud().generate(reviews)

#Display the generated image:
wordcloud = WordCloud(max_font_size=200).generate(reviews);
plt.figure();
plt.imshow(wordcloud, interpolation="bilinear");
plt.axis("off");
plt.show();


# __Analysis__: We tried to find out the top 10 most frequently words in the reviews, so that we can do further analysis.<br/> But as mentioned before, our scrapped data is not complete yet, so the result cannot give us much insight now.

# ## Compute summary statistics for all restaurants

# In[81]:


restaurant = df.groupby("Restaurant_Name")
restaurant.describe()


# ## Top 5 rated restaurants

# In[82]:


restaurant.mean().sort_values(by="Score", ascending=False).head()


# ##  Deal with Missing Values
# - Find variables with missing values
# - How to deal variables with missing values
#   - drop samples (rows)
#   - drop variables (columns)
#   - interpolate

# In[83]:


df.isnull().sum(axis=0)


# - __Analysis__: At this stage, we don't have any missing value in our scrapped data. We will remove any existing missing value once we get all the data because our dataset will be large enough to ignore these missing values.

# ## Remove Punctuations in reviews

# In[84]:


def text_process(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# ## Convert all reviews to lower case and Remove stop words

# In[85]:


# remove stop words
def text_process(text):
#     '''
#     Takes in a string of text, then performs the following:
#     1. Remove all punctuation
#     2. Remove all stopwords
#     3. Return the cleaned text as a list of words
#     '''
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[86]:


print(len(df["review"]))


# In[1]:


# # to lower case
# lowerReview = []

# for review in df["review"]:
#     lowerReview.append(text_process(review.lower()))
#     #print(lowerReview)
    


# In[ ]:


print(lowerReview[14])


# In[ ]:


df["lowerReview"] = lowerReview


# In[ ]:


df2 = df.drop(columns = ['review'], axis = 1)
print(df2.head)


# ## Convert text to sequence

# In[ ]:





# ## Convert Reviews to list of numbers using dictionary (!!! Currently Skipped)

# In[ ]:


reviewDict ={}


# In[ ]:


#print(type(df["Review"]))


# In[ ]:


# temporary lists of reviews
tests = df["lowerReview"].tolist()


# In[ ]:


print(type(tests))


# In[ ]:


# Convert all reviews to list of lists


# In[ ]:


from keras.preprocessing.text import Tokenizer, text_to_word_sequence
def converter(test):
    
    #train_docs = ['this is text number one', 'another text that i have']
    tknzr = Tokenizer(lower=True, split=" ")
    tknzr.fit_on_texts(test)
    #vocabulary:
    print(tknzr.word_index)


# In[ ]:


converter(tests[1])


# ## Convert all reviews to a single String to generate indexed review dictionary

# In[13]:


# put all the reviews into a single String
reviewCombined = []
for review in df["lowerReview"]:
    #print(type(review))
    #print(review[0])
    
    reviewCombined.append(review) #+= ' '.join(review)
print(reviewCombined[0:1000])


# In[14]:


from nltk import word_tokenize
# convert the reviews to dictionary

train_docs0 = reviewCombined    #['this is text number one', 'another text that i have']
#print(train_docs0[0])
train_docs = ""
for docs in train_docs0:
    train_docs += ' '.join(docs)
    print(docs)
# print(train_docs[0:15])
# print(type(train_docs[0]))


tokens = word_tokenize(train_docs)
#print(tokens[4])
   
#print(dict(enumerate(tokens))[0])
voc = {v: k for k, v in dict(enumerate(tokens)).items()}
print(type(voc))


# In[16]:


print(voc["wait"])


# In[17]:


# create a new column, representing the keys of words in each review 

reviewIndex = []



#print(df["Review"][0])
for review in reviewCombined:
    singleReviewIndex = []
    
    #print(review)
    
    
    for word in review:
        if word in voc.keys():
            singleReviewIndex.append(voc[word])
        
    reviewIndex.append(singleReviewIndex)
print(reviewIndex[0])


# In[18]:


df["indexedReview"] = reviewIndex


# In[27]:


print(df.head())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Generate balanced dataset from the original one

# In[88]:


countZeroOne = []
newCSV = []
countOnes = 0
for idx, i in enumerate(df["Score"]):
    if i == 5 or i ==4:
        countOnes += 1
        
        # select randomly 1836 ones and append them into newCSV
        if(countOnes%7 == 0):
            countZeroOne.append(1)
            newCSV.append([df["review"][idx],df["Sentiment"][idx]])
    elif i == 2 or i == 1:
        countZeroOne.append(0)
        # append all zeros into newCSV✅
        newCSV.append([df["review"][idx],df["Sentiment"][idx]])


# In[89]:


14314/1836


# In[90]:


zeros = countZeroOne.count(0)    # low scores: 1 & 2
ones = countZeroOne.count(1)     # high scores: 4 & 5
print(type(ones))
print(zeros)
print(ones)


# In[91]:


newCSV[0:3]


# In[74]:


newCsvDF = pd.DataFrame(newCSV)


# In[76]:


print(newCsvDF.head())


# In[78]:


# write simplified csv
import csv

with open("balancedData.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(newCSV)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Visualization: Analysis of rating distribution 

# In[20]:


ax=df.Score.value_counts().   plot.barh(figsize=(8,6), title="Rating distribution");
ax.set(ylabel="ratings", xlabel="number of restaurants");


# - __Analysis__: From the bar chart, we can see most restaurants are rated with full score, which indicates the overall popularity 
#     of the restaurants are relatively high.

# ## Tokenization for reviews

# In[32]:


df['tokenized_sents'] = df.apply(lambda column: nltk.word_tokenize(column['review']), axis=1)
df


# __Analysis__: At this stage, we can see the length of each review is quite long,<br/> we plan to extract top 10 most frequent words to do sentiment analysis in future.

# # Part 3 - Text Analysis

# ## General Info of the dataset

# In[29]:


# data = pd.read_csv('rawData.csv')
# print(type(data))
#print(data."Customer Name")
df.head()
df.info() # get detailed information of each column 
df.describe()
print( "\nThere are {} observations in this dataset. \n".format(df.shape[0]))


# ## Remove Stop Words

# In[30]:


# t0 = time()

# # Load spacy NLP object
# nlp = spacy.load('en_core_web_sm')

# # A list to store common words by all publications
# common_words = []

# # A dictionary to store the spacy_doc object of each publication
# publication_docs = {}

# reviews = df.Customer_Name.value_counts()[data.Customer_Name.value_counts()>6000][-10:].index.tolist()
# #names = data.publication.value_counts()[data.publication.value_counts()>6000][-10:].index.tolist()
# print(reviews[0:4])


# for review in reviews:
#     # Corpus is all the text written by that publication
#     corpus = ""
#     # Grab all rows of current publication, along the 'content' column
#     review_contents = df.loc[df.Customer_Name==name,'Review']
    
#     # Merge all articles in to the publication's corpus
#     for review_content in review_contents:
#         corpus += review_content
#     # Let Spacy parse the publication's body of text
#     doc = nlp(corpus)
    
#     # Store the doc in the dictionary
#     publication_docs[name] = doc
        
#     # Filter out punctuation and stop words.
#     lemmas = [token.lemma_ for token in doc
#                 if not token.is_punct and not token.is_stop]
        
#     # Return the most common words of that publication's corpus.
#     words = [item[0] for item in Counter(lemmas).most_common(1000)]
    
#     # Add them to the list of words by all publications.
#     for word in words:
#         common_words.append(word)

# # Eliminate duplicates
# common_words = set(common_words)
    
# print('Total number of common words:',len(common_words))
# print("done in %0.3fs" % (time() - t0))


# In[31]:


# # step 3. get document-term matrix
# # contruct a document-term matrix where 
# # each row is a doc 
# # each column is a token
# # and the value is the frequency of the token
# import pandas as pd
# # since we have a small corpus, we can use dataframe 
# # to get document-term matrix
# # but don't use this when you have a large corpus
# dtm=pd.DataFrame.from_dict(docs_tokens, \
#                            orient="index" )
# dtm=dtm.fillna(0)
# dtm


# In[32]:


# # step 4. get normalized term frequency (tf) matrix
# # convert dtm to numpy arrays
# tf=dtm.values
# #print(tf)
# # sum the value of each row
# doc_len=tf.sum(axis=1)
# print(doc_len)
# # divide dtm matrix by the doc length matrix
# tf=np.divide(tf, doc_len[:,None])
# print(tf[6:10])


# In[33]:


# # step 5. get idf
# # get document freqent
# df=np.where(tf>0,1,0)
# #df
# # get idf
# idf=np.log(np.divide(len(reviews), \
#         np.sum(df, axis=0)))+1
# print("\nIDF Matrix")
# print (idf)
# smoothed_idf=np.log(np.divide(len(docs)+1, np.sum(df, axis=0)+1))+1
# print("\nSmoothed IDF Matrix")
# print(smoothed_idf)


# 

# ## Positive words in each comment

# In[34]:


df['tokenized_sents'] = df.apply(lambda column: nltk.word_tokenize(column['Review']), axis=1)


# In[34]:


# Positive Words in comments
commentPositiveWordsList = []
#print(df['tokenized_sents'])
reviewLists = df['lowerReview'].tolist()
print(reviewLists[0:1])
for idx, review in enumerate(reviewLists):
    with open("positive-words.txt",'r') as f:
        positive_words=[line.strip() for line in f]
    #print(df["Score"][idx])
    #positive_words
    #print(positive_words)
    
    positive_tokens=[token for token in review                      if token in positive_words]
    #print(positive_tokens)
    commentPositiveWordsList.append(positive_tokens)


# ## Negative words in each comment

# In[35]:


# negative Words in comments
commentNegativeWordsList = []
for text in df['tokenized_sents']:
    with open("negative-words.txt",'r') as f:
        negative_words=[line.strip() for line in f]
    #positive_words
    #print(positive_words)
    negative_tokens=[token for token in text                      if token in negative_words]
    #print(negative_tokens)
    commentNegativeWordsList.append(negative_tokens)


# ## Remove the positive words is preceeded by negation words

# In[37]:


# Exercise 3.5.2.2 # check if a positive word is preceded by negation words
# e.g. not, too, n't, no, cannot
# this is not an exhaustive list of negation words!
negations=['not', 'too', 'n\'t', 'no', 'cannot', 'neither','nor']
true_positive_tokens=[]

for text in df['tokenized_sents']:
    for idx, token in enumerate(text):
        if token in positive_words:
            if idx>0:
                if text[idx-1] not in negations:
                    positive_tokens.append(token)
            else:
                positive_tokens.append(token)
    # compare the positive_tokens generated above with poitive 
    
    print(positive_tokens)


# In[ ]:





# In[ ]:





# # Topic Modeling

# In[36]:


import json
from numpy.random import shuffle

reviews = df['tokenized_sents']
print(reviews.head())
reviewsArray = reviews.values
print(reviewsArray)


shuffle(reviewsArray.tolist())


# In[37]:


print(type(reviews))
reviews_df = pd.DataFrame(reviews)
#print(reviews_df.head)


# In[15]:


# Exercise 4.2. Preprocessing - Create Term Frequency Matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# LDA can only use raw term counts for LDA 
tf_vectorizer = CountVectorizer(max_df=0.90,                 min_df=50, stop_words='english')

# reviews = df['tokenized_sents']
print(type(reviewsArray))
print(reviews[0:2])

arrayList = []

for array in reviewsArray:
    for word in array:
        arrayList.append(word)
        #print(word)

print(arrayList[0])        
        
#reviews_df = pd.DataFrame(reviewsArray)
tf = tf_vectorizer.fit_transform(arrayList)
# each feature is a word (bag of words)
# get_feature_names() gives all words
tf_feature_names = tf_vectorizer.get_feature_names()

print(tf_feature_names[0:10])
print(tf.shape)

# split dataset into train (90%) and test sets (10%)
# the test sets will be used to evaluate proplexity of topic modeling
X_train, X_test = train_test_split(                tf, test_size=0.1, random_state=0)


# ## Train LDA model

# In[39]:


# 
from sklearn.decomposition import LatentDirichletAllocation

num_topics = 4

# Run LDA. For details, check
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html#sklearn.decomposition.LatentDirichletAllocation.perplexity

# max_Hongyi Cheniter control the number of iterations 
# evaluate_every determines how often the perplexity is calculated
# n_jobs is the number of parallel threads
lda = LatentDirichletAllocation(n_components=num_topics,                                 max_iter=5,verbose=1,
                                evaluate_every=1, n_jobs=1,
                                random_state=0).fit(X_train)


# ## Check topic and word distribution per topic

# In[40]:


# Exercise 4.4. Check topic and word distribution per topic

num_top_words=20

# lda.components_ returns a KxN matrix
# for word distribution in each topic.
# Each row consists of 
# probability (counts) of each word in the feature space

for topic_idx, topic in enumerate(lda.components_):
    print ("Topic %d:" % (topic_idx))
    # print out top 20 words per topic 
    words=[(tf_feature_names[i],topic[i])            for i in topic.argsort()[::-1][0:num_top_words]]
    print(words)
    print("\n")


# ## Plot Wordcloud

# In[41]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud
import math

num_top_words=50
f, axarr = plt.subplots(2, 2, figsize=(8, 8));

for topic_idx, topic in enumerate(lda.components_):
    # create a dataframe with two columns (word, weight) for each topic
    
    # create a word:count dictionary
    f={tf_feature_names[i]:topic[i] for i in topic.argsort()[::-1][0:num_top_words]}

    # generate wordcloud in subplots
    wordcloud = WordCloud(width=480, height=450, margin=0, background_color="black");
    _ = wordcloud.generate_from_frequencies(frequencies=f);
    
    _ = axarr[math.floor(topic_idx/2), topic_idx%2].imshow(wordcloud, interpolation="bilinear");
    _ = axarr[math.floor(topic_idx/2), topic_idx%2].set_title("Topic: "+str(topic_idx));
    _ = axarr[math.floor(topic_idx/2), topic_idx%2].axis('off')

plt.tight_layout()
plt.show()


# # 1.  把wordcloud中前四个词作为四个feature，去count每一条评论中各词出现次数

# In[44]:


# # Automatic select the 4 
# for topic_idx, topic in enumerate(lda.components_):
#     f={tf_feature_names[i]:topic[i] for i in topic.argsort()[::-1][0:num_top_words]}
#     print(f.)
#     #print(str(topic_idx))
#     #print(str(topic))


# In[14]:


sLength = len(df['Score'])
count0 = []
count1 = []
count2 = []
count3 = []
#delete the created column from last run
# del df['service']
# del df['time']
# del df['place']
# del df['wait']
for idx in range(0,sLength): 
#     print(type(df['review'][idx]))
    #print(sum(df['review'][idx].count(x) for x in ["time", "food", "pretty"]))
    #print(df['review'][1])
    
    # Count the occurence of top 3 words in each topic for each review
    
    count0.append(sum(df['review'][idx].count(x) for x in ("time", "food", "pretty")))
    #print(count0)
    count1.append(sum(df['review'][idx].count(x) for x in ("wait", "good", "experience")))
    count2.append(sum(df['review'][idx].count(x) for x in ("ordered", "table", "great")))
    count3.append(sum(df['review'][idx].count(x) for x in ("place", "service", "like")))
    #print(count1)
    #print(count2)
    #print(count3)
print(pd.Series(count0, index = df.index)[2])    
# df['time, food, pretty'] = x for x in pd.Series(count0, index=df.index)
# df['wait, good, experience'] = pd.Series(count1, index=df.index)
# df['ordered, table, great'] = pd.Series(count2, index=df.index)
# df['place, service, like'] = pd.Series(count3, index=df.index)

# #print(data.head)
# df.to_csv("processedData.csv")


# ## Assign documents to topic

# In[71]:


import numpy as np

# Generate topic assignment of each document
topic_assign=lda.transform(X_train)

print(topic_assign[0:5])

# set a probability threshold
# the threshold determines precision/recall
prob_threshold=0.25

topics=np.copy(topic_assign)
topics=np.where(topics>=prob_threshold, 1, 0)
print(topics[0:5])


# ## Evaluate topic models by perplexity of test data

# In[72]:


perplexity=lda.perplexity(X_test)
print(perplexity)


# ## Best number of topics (Needs modification)

# In[73]:


import numpy as np
import matplotlib.pyplot as plt

result=[]
for num_topics in range(2,5):
    lda = LatentDirichletAllocation(n_components=num_topics,                                 learning_method='online',                                 max_iter=10,verbose=0, n_jobs=1,
                                random_state=0).fit(X_train)
    p=lda.perplexity(X_test)
    result.append([num_topics,p])
    print(num_topics, p)


# In[ ]:


# pd.DataFrame(result, columns=["K", "Perlexity"]).plot.line(x='K',y="Perlexity");
# plt.show();


# In[ ]:





# In[ ]:





# In[ ]:





# # 2. Review Clustering

# In[74]:


from sklearn.model_selection import train_test_split
import pandas as pd


# ## TF-IDF & NLTK

# In[78]:


# A subset is loaded

import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("raw_data_0504.csv",               header=0)

# Select three labels for now
labels =['chicken', 'pretty',         'service', 'menu']
#data=df[data['Review'][idx].count(x) for x in ("service", "Service", "SERVICE")]

# Split dataset into training and test. 
# Assuming we only know ground-truth label 
# for the test set.

train, test = train_test_split(df, test_size=0.2, random_state=0)

# print out the full text of the first sample
# print(data["Review"][0])


# In[6]:


# initialize the TfidfVectorizer 
# set min document frequency to 5

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.corpus import stopwords

# set the min document frequency to 5
# generate tfidf matrix
print(type(df["review"]))
tfidf_vect = TfidfVectorizer(stop_words="english",                             min_df=5) 

dtm= tfidf_vect.fit_transform(df["review"])
print (dtm.shape)

# set number of clusters
num_clusters=3

# initialize clustering model
# using cosine distance
# clustering will repeat 20 times
# each with different initial centroids
clusterer = KMeansClusterer(num_clusters,                             cosine_distance,                             repeats=20)

# samples are assigned to cluster labels 
# starting from 0
clusters = clusterer.cluster(dtm.toarray(),                              assign_clusters=True)

#print the cluster labels of the first 5 samples
print(clusters[0:5])





# df=pd.read_csv("rawData.csv",\
#                header=0)

# # Select 4 labels for now
# labels =['chicken', 'pretty',\
#          'service', "menu"]
# # print()

# # data=pd.DataFrame()
    
# #     if 'chicken' in enumerate(df["tokenized_sents"]):
# #         data.append()

# # print(data)

# df['tokenized_sents'] = df.apply(lambda column: nltk.word_tokenize(column['Review']), axis=1)

# # set the min document frequency to 5
# # generate tfidf matrix
# print(type(df['tokenized_sents']))harles
# tfidf_vect = TfidfVectorizer(stop_words="english",\
#                              min_df=5) 

# tfidf_vect.fit(df['tokenized_sents'])

# dtm= tfidf_vect.transform(df['tokenized_sents'])
# print (dtm.shape)


# In[82]:


# clusterer.means() contains the centroids
# each row is a cluster, and 
# each column is a feature (word)
centroids=np.array(clusterer.means())

# argsort sort the matrix in ascending order 
# and return locations of features before sorting
# [:,::-1] reverse the order
sorted_centroids = centroids.argsort()[:, ::-1] 

# The mapping between feature (word)
# index and feature (word) can be obtained by
# the vectorizer's function get_feature_names()
voc_lookup= tfidf_vect.get_feature_names()

for i in range(num_clusters):
    
    # get words with top 20 tf-idf weight in the centroid
    top_words=[voc_lookup[word_index]                for word_index in sorted_centroids[i, :20]]
    print("Cluster %d:\n %s " % (i, "; ".join(top_words)))


# ## Evaluate clusters

# In[7]:


# Exercise 5.2.1 Predict labels for new samples

# Question: how to determine 
# the label for a new sample?

# note transform function is used
# not fit_transform
test_dtm = tfidf_vect.transform(df["review"])

predicted = [clusterer.classify(v) for v in test_dtm.toarray()]

predicted[0:10]


# In[87]:


# Exercise 5.2.2 External evaluation
# determine cluster labels and calcuate precision and recall

# Create a dataframe with cluster id and 
# ground truth label
confusion_df = pd.DataFrame(list(zip(df["review"].values, predicted)),                            columns = ["review", "cluster"])
confusion_df.head()

# generate crosstab between clusters and true labels
pd.crosstab( index=confusion_df.cluster, columns=confusion_df.review)


# In[88]:


# Exercise 5.2.3 
# Map cluster id to true labels by "majority vote"
cluster_dict={0:'service',              1:"menu",              2:'pretty'}

# Map true label to cluster id
predicted_target=[cluster_dict[i] for i in predicted]

print(metrics.classification_report      (df["review"], predicted_target))


# In[89]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats
import seaborn as sns


# In[ ]:





# In[ ]:





# In[90]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.corpus import stopwords

# set the min document frequency to 5
# generate tfidf matrix
tfidf_vect = TfidfVectorizer(stop_words="english",                             min_df=5) 

dtm= tfidf_vect.fit_transform(arrayList)
print (dtm.shape)


# In[ ]:


# Exercise 5.1.3 Clustering using NLTK KMean
# cosine distance is calculated

from nltk.cluster import KMeansClusterer, cosine_distance

# set number of clusters
num_clusters=3

# initialize clustering model
# using cosine distance
# clustering will repeat 20 times
# each with different initial centroids
clusterer = KMeansClusterer(num_clusters,                             cosine_distance,                             repeats=20)

# samples are assigned to cluster labels 
# starting from 0
clusters = clusterer.cluster(dtm.toarray(),                              assign_clusters=True)

#print the cluster labels of the first 5 samples
print(clusters[0:5])


# ## Top 10 Rating every month / 10 days/ week

# In[1]:


# print(df["Comment Date"].head)
months = {}
days = {}
years = {}

for idx,date in enumerate(df["review_date"]):
    #print(date[0])
    month, day, year = date.split("/")
    # form months dictionary
    if month in months:
        months[month]+=1
    else:
        months[month]=1
    # form day dictionary
    if day in days:
        days[day]+=1
    else:
        days[day]=1
    # form years dictionary
    if year in years:
        years[year]+=1
    else:
        years[year]=1
    
print(months)
print("\n", days)
print("\n",years)


# In[ ]:





# ## Prediction of ......

# In[ ]:


# Establish outcome and predictors
y = df['tokenized_sents'][0:100]
X = df['tokenized_sents'][101:201]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


y_test.value_counts()


# # Modeling

# In[8]:


df = pd.read_csv('processedData.csv')
df.head()
df.info() # get detailed information of each column 
df.describe()


# In[9]:


# Seperate the dataset into X and Y for prediction
x = df['review']
y = df['Score']
print(x.head())

print(y.head())


# In[34]:


# Exercise 3.2 Create TF-IDF Matrix

from sklearn.feature_extraction.text import TfidfVectorizer

# initialize the TfidfVectorizer 
tfidf_vect = TfidfVectorizer() 

# with stop words removed
tfidf_vect = TfidfVectorizer(stop_words="english") 

# generate tfidf matrix
dtm= tfidf_vect.fit_transform(x)
#print(dtm)
print("type of dtm:", type(dtm))
print("size of tfidf matrix:", dtm.shape)


# In[35]:


# Exercise 3.3. Examine TF-IDF

# 1. Check vocabulary

# Vocabulary is a dictionary mapping a word to an index

# the number of words in the vocabulary
print("total number of words:", len(tfidf_vect.vocabulary_))

print("type of vocabulary:",       type(tfidf_vect.vocabulary_))
print("index of word 'city' in vocabulary:",       tfidf_vect.vocabulary_['korean'])


# In[36]:


# 3.4 check words with top tf-idf wights in a document, 
# e.g. 1st document

# get mapping from word index to word
# i.e. reversal mapping of tfidf_vect.vocabulary_
voc_lookup={tfidf_vect.vocabulary_[word]:word             for word in tfidf_vect.vocabulary_}

print("\nOriginal text: \n"+x[1])

print("\ntfidf weights: \n")

# first, covert the sparse matrix row to a dense array
doc0=dtm[0].toarray()[0]
print(doc0.shape)

# get index of top 20 words
top_words=(doc0.argsort())[::-1][0:20]
[(voc_lookup[i], doc0[i]) for i in top_words]



# In[38]:


# Exercise 3.5. classification using a single fold

# use MultinomialNB algorithm
from sklearn.naive_bayes import MultinomialNB

# import method for split train/test data set
from sklearn.model_selection import train_test_split

# import method to calculate metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

# split dataset into train (70%) and test sets (30%)
X_train, X_test, y_train, y_test = train_test_split(                dtm, x, test_size=0.3, random_state=0)
print(X_train.shape[0])
#print(type(X_train))
# train a multinomial naive Bayes model using the testing data
clf = MultinomialNB().fit(X_train, y_train)

# predict the news group for the test dataset
predicted=clf.predict(X_test)

# check a few samples
predicted[0:3]
y_test[0:3]


# In[42]:


print(type(predicted))


# In[39]:


# n_columns = "review"
# dummy_df = pd.get_dummies(data[n_columns])# 用get_dummies进行one hot编码
# data = pd.concat([data, dummy_df], axis=1)


# In[74]:


labels = df["Restaurant_Name"][0:4]
print(type(labels))
labels.tolist()


# In[67]:


labels = sorted((labels.unique()))


# In[72]:


print(type(labels[0]))
print(predicted[0])
print(len(labels))
#print(type(y_test[0]))


# In[73]:


# Exercise 3.6. Performance evaluation: 
# precision, recall, f1-score

# get the list of unique labels
#labels=sorted(X_train.unique())

# calculate performance metrics. 
# Support is the number of occurrences of each label
#print(type(y))
      
precision, recall, fscore, support=     precision_recall_fscore_support(     y_test, predicted, labels=labels)

print("labels: ", labels)
print("precision: ", precision)
print("recall: ", recall)
print("f-score: ", fscore)
print("support: ", support)

# another way to get all performance metrics
print(classification_report(x, y, target_names=labels))


# In[ ]:


# Exercise 3.7.  AUC 

from sklearn.metrics import roc_curve, auc,precision_recall_curve
import numpy as np

# We need to get probabilities as predictions
predict_p=clf.predict_proba(X_test)

# a probability is generated for each label
# labels
# predict_p[0:3]
# # Ground-truth
# y_test[0:3]

# let's just look at one label "soc.religion.christian"
# convert to binary
binary_y = np.where(y_test=="service test recall wait for too long",1,0)

# this label corresponds to last column
y_pred = predict_p[:,3]

# compute fpr/tpr by different thresholds
# positive class has label "1"
fpr, tpr, thresholds = roc_curve(binary_y, y_pred,                                  pos_label=1)
print(fpr)
# calculate auc
print(auc(fpr, tpr))


# In[ ]:





# In[ ]:





# ## SVM

# In[76]:


# # Support Vector Machine
# import os
# import numpy as np
# from collections import Counter
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# def make_Dictionary(root_dir):
#     all_words = []
#     emails = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
#     for mail in emails:
#         with open(mail) as m:
#             for line in m:
#                 words = line.split()
#                 all_words += words
#     dictionary = Counter(all_words)
#     list_to_remove = list(dictionary)
    
# for item in list_to_remove:
#         if item.isalpha() == False:
#             del dictionary[item]
#         elif len(item) == 1:
#             del dictionary[item]
#     dictionary = dictionary.most_common(3000)
# return dictionary

# def extract_features(mail_dir):
#     files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
#     features_matrix = np.zeros((len(files),3000))
#     train_labels = np.zeros(len(files))
#     count = 0;
#     docID = 0;
#     for fil in files:
#         with open(fil) as fi:
#             for i,line in enumerate(fi):
#                 if i == 2:
#                     words = line.split()
#                     for word in words:
#                         wordID = 0
#                           for i,d in enumerate(dictionary):
#                                 if d[0] == word:
#                                     wordID = i
#                                     features_matrix[docID,wordID] = words.count(word)
#             train_labels[docID] = 0;
#             filepathTokens = fil.split('/')
#             lastToken = filepathTokens[len(filepathTokens) - 1]
#             if lastToken.startswith("spmsg"):
#                 train_labels[docID] = 1;
#                 count = count + 1
#             docID = docID + 1
#     return features_matrix, train_labels

# SVM




# dictionary = voc

# # import method for split train/test data set
# from sklearn.model_selection import train_test_split

# # import method to calculate metrics
# from sklearn.metrics import precision_recall_fscore_support
# from sklearn.metrics import classification_report

# # split dataset into train (70%) and test sets (30%)
# X_train, X_test, y_train, y_test = train_test_split(\
#                 dtm, data["label"], test_size=0.3, random_state=0)

# print "reading and processing emails from file."
# features_matrix, labels = extract_features(TRAIN_DIR)
# test_feature_matrix, test_labels = extract_features(TEST_DIR)
# model = svm.SVC()
# print "Training model."
# #train model
# model.fit(features_matrix, labels)
# predicted_labels = model.predict(test_feature_matrix)
# print "FINISHED classifying. accuracy score : "
# print accuracy_score(test_labels, predicted_labels)

from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=1E10)
model.fit(x, y)


# In[ ]:





# In[ ]:




