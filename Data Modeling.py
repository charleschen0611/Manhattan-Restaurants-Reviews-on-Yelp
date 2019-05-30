#!/usr/bin/env python
# coding: utf-8

# # Project topic: Restaurant Popularity Analysis Using Yelp Reviews<br/>
# ## Part 4 - Modeling<br/>
# - BIA-660-Group 7<br/>
# - Instructor: Prof. Rong Liu<br/>
# - Group member: Honyi Chen, Tingyi Lu, Junhan Zhou, Xiaomin Yang<br/>
# - 2019 Spring

# In[94]:


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
from nltk.corpus import stopwords
import string

#import shuffle


# # Load the Balanced Data

# In[95]:


data = pd.read_csv('balancedData.csv',header=0)
data.head()
# rawData = pd.read_csv('rawData.csv',sep='delimiter')
# rawData.head()
# data.info() # get detailed information of each column 
df= pd.DataFrame(data)
df = df.sample(frac = 1).reset_index(drop = True)
df
#df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)


# 

# In[96]:


# Create X and Y for classification tasks
# X will be the review column of data_class, and y will be the Sentiment column.

X=df['review']
Y=df['Sentiment']


# In[97]:


X[0]


# In[98]:


import string
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


# In[99]:


#text the function
sample_text = "Hey there! This is a sample review, which happens to contain punctuations."
print(text_process(sample_text))


# In[100]:


from sklearn.feature_extraction.text import CountVectorizer
#import CountVectorizer and fit an instance to our review text (stored in X), 
#passing in our text_process function as the analyser.
bow_transformer = CountVectorizer(analyzer=text_process).fit(X)


# In[101]:


#size of the vocabulary stored in teh vectoriser
len(bow_transformer.vocabulary_)


# In[102]:


#transform X df into a sparse matrix
X = bow_transformer.transform(X)


# In[103]:


print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)
# Percentage of non-zero values
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
print("Density: {}".format((density)))


# As we have finished processing the review text in X, It’s time to split our X and Y into a training and a test set using train_test_split from Scikit-learn. We will use 30% of the dataset for testing.

# In[147]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


# In[148]:


X_train


# In[149]:


Y_train=np.int64(Y_train)
Y_test=np.int64(Y_test)


# # Multinomial Naive Bayes

# Multinomial Naive Bayes is a specialised version of Naive Bayes designed more for text documents. Let’s build a Multinomial Naive Bayes model and fit it to our training set (X_train and y_train).

# In[165]:


from scipy import interp
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


# In[167]:


#train model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
y_score1=nb.fit(X_train, Y_train)


# Our model has now been trained! It’s time to see how well it predicts the ratings of previously unseen reviews (reviews from the test set). First, let’s store the predictions as a separate dataframe called preds.

# In[151]:


y_pred1 = nb.predict(X_test)


# In[152]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test, y_pred1))
print('\n')
print(classification_report(Y_test, y_pred1))


# In[153]:


from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:





# In[154]:


nb_cv_score = cross_val_score(nb,X, Y, cv = 10, scoring = "roc_auc")


# In[155]:


print("=== Confusion Matrix ===")
print(confusion_matrix(Y_test, y_pred1))
print('\n')
print("=== Classification Report ===")
print(classification_report(Y_test, y_pred1))
print('\n')
print("=== All AUC Scores ===")
print(nb_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Naive Bayes: ", nb_cv_score.mean())


# ### Print AUC Curve

# In[188]:


from matplotlib import pyplot as plt
plt.figure();
plt.plot(fpr, tpr, color='darkorange', lw=2);
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('AUC of Naive Bayes Model');
plt.show();


# ## Support Vector Machine

# In[23]:


from sklearn.svm import LinearSVC
svc = LinearSVC()


# In[24]:


svc.fit(X_train,Y_train)


# In[25]:


y_pred2 = svc.predict(X_test)


# In[26]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test, y_pred2))
print('\n')
print(classification_report(Y_test, y_pred2))


# In[27]:


svc_cv_score = cross_val_score(svc,X, Y, cv = 10, scoring = "roc_auc")


# In[28]:


print("=== Confusion Matrix ===")
print(confusion_matrix(Y_test, y_pred2))
print('\n')
print("=== Classification Report ===")
print(classification_report(Y_test, y_pred2))
print('\n')
print("=== All AUC Scores ===")
print(svc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - SVM: ", svc_cv_score.mean())


# In[180]:


plt.figure();
plt.plot(fpr, tpr, color='darkorange', lw=2);
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--');
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate');
plt.ylabel('True Positive Rate');
plt.title('AUC of Naive Bayes Model');
plt.show();


# # KNN

# In[181]:


#Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier

#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)

#Train the model using the training sets
knn.fit(X_train, Y_train)

#Predict the response for test dataset
y_pred3 = knn.predict(X_test)


# In[182]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred3))


# In[183]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(Y_test, y_pred3))
print('\n')
print(classification_report(Y_test, y_pred3))


# In[184]:


knn_cv_score = cross_val_score(knn,X, Y, cv = 10, scoring = "roc_auc")


# In[185]:


print("=== Confusion Matrix ===")
print(confusion_matrix(Y_test, y_pred3))
print('\n')
print("=== Classification Report ===")
print(classification_report(Y_test, y_pred3))
print('\n')
print("=== All AUC Scores ===")
print(knn_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - KNN: ", knn_cv_score.mean())


# ## Random Forest

# In[34]:


from sklearn.ensemble import RandomForestClassifier


# In[35]:


from sklearn import model_selection

# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)
# predictions
rfc_predict = rfc.predict(X_test)


# In[36]:


from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report, confusion_matrix


# In[37]:


rfc_cv_score = cross_val_score(rfc, X, Y, cv=10, scoring='roc_auc')


# In[38]:


print("=== Confusion Matrix ===")
print(confusion_matrix(Y_test, rfc_predict))
print('\n')
print("=== Classification Report ===")
print(classification_report(Y_test, rfc_predict))
print('\n')
print("=== All AUC Scores ===")
print(rfc_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())


# ## Decision Tree

# In[39]:


from sklearn.tree import DecisionTreeClassifier


# In[40]:


# Create Decision Tree classifer object
dt = DecisionTreeClassifier()

# Train Decision Tree Classifer
dt = dt.fit(X_train,Y_train)

#Predict the response for test dataset
y_pred5 = dt.predict(X_test)


# In[41]:


dt_cv_score = cross_val_score(dt, X, Y, cv=10, scoring='roc_auc')


# In[42]:


print("=== Confusion Matrix ===")
print(confusion_matrix(Y_test, y_pred5))
print('\n')
print("=== Classification Report ===")
print(classification_report(Y_test, y_pred5))
print('\n')
print("=== All AUC Scores ===")
print(dt_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Decision Tree: ", dt_cv_score.mean())


# ## Logistic Regression

# In[43]:


# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
lr = LogisticRegression()

# fit the model with data
lr.fit(X_train,Y_train)

#
y_pred6=lr.predict(X_test)


# In[44]:


lr_cv_score = cross_val_score(lr, X, Y, cv=10, scoring='roc_auc')


# In[45]:


print("=== Confusion Matrix ===")
print(confusion_matrix(Y_test, y_pred6))
print('\n')
print("=== Classification Report ===")
print(classification_report(Y_test, y_pred6))
print('\n')
print("=== All AUC Scores ===")
print(lr_cv_score)
print('\n')
print("=== Mean AUC Score ===")
print("Mean AUC Score - Logistic Regression: ", lr_cv_score.mean())


# In[ ]:


from sklearn.svm import SVC # "Support vector classifier"
model = SVC(kernel='linear', C=1E10)
model.fit(x, z)


# In[ ]:





# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier


clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1, n_splits = 2, 
     random_state=0)
scores = cross_val_score(clf, binary_y, y_pred, cv=5)
print(scores.mean())                            

clf = RandomForestClassifier(n_estimators=10, max_depth=None,
     min_samples_split=2, random_state=0)
scores = cross_val_score(clf, binary_y, y_pred, cv=5)
print(scores.mean())                               

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
scores = cross_val_score(clf, binary_y, y_pred, cv=5)
print(scores.mean())


# In[ ]:





# # Future Work: Loistic最高，所以最合适，因为LR是最好的，未来可以做更多研究（Feature）

# In[ ]:





# ## Clustering

# # 下边都删掉

# In[48]:


# # Data Scaling
# ss = StandardScaler()
# ss.fit_transform(df["lowerReview"])


# In[ ]:





# In[49]:


review2DArray = []


# In[52]:


for array in df["review"]:
    review2DArray.append(array)


# In[53]:


print(review2DArray[0:3])


# In[ ]:





# In[55]:


# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

print(X)
print(Y)
forest.fit(X, Y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[ ]:





# In[ ]:




