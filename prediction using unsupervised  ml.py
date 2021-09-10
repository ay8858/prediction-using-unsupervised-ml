#!/usr/bin/env python
# coding: utf-8

# AUTHOR: ASHISH YADAV
# 
# TECHNICAL TASK: PREDICTION USING UNSUPERVISED (LEVEL-BEGNNIER)
# 
# In this task,We are going to predict the optimum number of clusters from the given iris dataset and represent it visually unsupervised learning.

# In[1]:


#Importing all the library which we needed in this notebook

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets


# 
# Step 1: Reading the data-set
# 

# In[2]:


#loading and reading the iris dataset
#data will get from this link:- "https://bit.ly/3kXTdox"

data=pd.read_csv('F:\Dataset/iris.csv')
print('Data successfully loaded')


# In[3]:


print(data.head()) #Load the first five row given below


# In[4]:


print(data.tail()) #Load the last five row given below


# In[5]:


#Checking For NAN value

print(data.isna().sum())


# NaN standing for Not a Number, is a member of a numeric data type that can be interpreted as a value that is undefined or unrepresentable, especially in floating-point arithmetic.For example 0/0 is undefined as real number and is,therefore,represented by Nan. So in this dataset,we dont have such value. 

# In[9]:


#checking for statistical description
print(data.describe())


# Now, let's check unique classes in the dataset.

# In[17]:


print(data.Species.nunique())
print(data.Species.value_counts())


# Step 2 - Data Visualization

# In[19]:


sns.set(style='whitegrid')
iris=sns.load_dataset('iris');
ax=sns.stripplot(x='species',y='sepal_length',data=iris);
plt.title('Iris Dataset')
plt.show()


# In[20]:


sns.boxplot(x='species',y='sepal_length',data=iris);
plt.title('Iris Dataset')
plt.show()


# In[21]:


sns.boxplot(x='species',y='petal_width',data=iris);
plt.title('Iris Dataset')
plt.show()


# In[25]:


sns.boxplot(x='species',y='petal_length',data=iris);
plt.title('Iris Dataset')
plt.show()


# In[28]:


#count plot
sns.countplot(x='species',data=iris, palette='OrRd')
plt.title('count of different species in Irish datasheet' )
plt.show()


# In[29]:


#heat map
sns.heatmap(data.corr(), annot=True,cmap='RdYlGn')
plt.title('Heat Map' )
plt.show()


# Step 3 - Find the optimum number of clusters using K-means clustering

# In[38]:


#Find the optimum number of clusters using K-means

x=data.iloc[:,[0,1,2,3]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    ##apeanding the wcss to the list (k-means.intertia_return the WCSS value for an initialize cluster)
    wcss.append(kmeans.inertia_)
    print('k:',i, "wcss:",kmeans.inertia_)


# In[39]:


#Plotting the result onto a line graph,allowing us to observe 'The eblow'

plt.plot(range(1,11),wcss)
plt.title('The Elbow method')
plt.xlabel('number of cluster')
plt.ylabel('WCSS')
plt.show()


# We can see that after 3 the drop in wcss is minimal.So we choose the 3 as a optimal number of clusters.

# 
# Step 4 - Initializing K-Means with optimum number of clusters 

# In[40]:


#Fitting K-Means to the Dataset
kmeans=KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

#Return a label for each data point based on the number of clusters
y_kmeans=kmeans.fit_predict(x)


# Step 5 - Predicting values

# In[41]:


y_kmeans


# In[43]:


#visualising the clusters

plt.figure(figsize=(10,10))
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label='Iris-versicolour')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='Iris-verginica')

#plotting the centroids of the cluster

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c="yellow",label='centroids')
plt.title('Iris Flower Clusters')
plt.xlabel('sepal length in cm')
plt.ylabel('petal length in cm')
plt.legend()
plt.show()


# In[ ]:




