#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries 

# In[2]:


import numpy as np
import pandas as pd


# ## Load datasets

# In[3]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies.shape


# In[5]:


movies.head(3)


# In[6]:


credits.shape


# In[7]:


credits.head()


# In[8]:


#merge both dataframe on the basis of title column 
movies=pd.merge(movies,credits,on='title')


# In[9]:


movies.shape


# In[10]:


movies.head()


# In[11]:


# we'll choose those columns which will help in creating tags
#genres
#id
#keywords (they are basically tags)
#title 
#language not needed (as approx. 95% lang. are in eng)
#original title not needed (as we've alredy taken title)
#overview
#popularity,production_companies,	production_countrie,release date,revenue.... not needed (also removing numeric features)
#cast , crew 


# In[12]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[13]:


movies.head(3)


# ### Missing Values

# In[14]:


movies.isnull().sum()


# In[15]:


movies.dropna(inplace=True)


# In[16]:


movies.isnull().sum()


# In[17]:


# check duplicate data
movies.duplicated().sum()


# In[18]:


movies.iloc[0].genres


# ## Preprocessing

# In[20]:


# Convert '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
# into ['Actuon','Adventure','Fantasy','SciFi']


# In[21]:


import ast
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[22]:


movies['genres']=movies['genres'].apply(convert)


# In[23]:


movies.head(3)


# In[24]:


movies.iloc[0].keywords


# In[25]:


movies['keywords']=movies['keywords'].apply(convert)


# In[26]:


movies.head(3)


# In[27]:


movies.iloc[0].cast  


# In[28]:


# But we want only first three actors


# In[29]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[30]:


movies['cast'].apply(convert3)


# In[31]:


movies['cast']=movies['cast'].apply(convert3)


# In[32]:


movies.head(2)


# In[33]:


movies['crew'][0]


# In[34]:


# we just want that dic value in the crew feature in which job value is director


# In[35]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[36]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[37]:


movies.head(2)


# In[38]:


movies['overview'][0]


# In[39]:


# converting this feature having string into list 
movies['overview'].apply(lambda x:x.split())


# In[40]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[41]:


movies.head(1)


# In[42]:


# now we have everything in the list (for each feature) , now we can easily concate these list to create a tag feature 
#one issue : removing the space in between two words like converting Sam Worthington into SamWorthington


# In[43]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x]) # to remove the space in genres
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[44]:


movies.head()


# In[45]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[46]:


movies.head(2)


# In[47]:


#now we have a feature tags having the values by concating some features, time to remove those features


# In[48]:


new_df=movies[['movie_id','title','tags']]


# In[49]:


new_df


# In[50]:


# want to convert list into string (in the column tags)


# In[51]:


new_df['tags'].apply(lambda x:" ".join(x))


# In[52]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[53]:


new_df.head(3)


# In[54]:


new_df['tags'][0]


# In[55]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[56]:


new_df.head()


# ## Vectorization

# In[58]:


# task: convert each movie into a vector
# we'll recommend those movies which are closest to that corresponding movie vector
# large text= tag1+ tag2...+tag 5k
# find w1,w2,.....,wn with increasing frequencies from large text
# corresponding to movie m1 , we'll assign the no to w1,w2,...wn whenever exists
#we have a matrix of order 5k x n
# each row will be the vetor and here we have n-dimension

# count vectorizer catches how often a word in a df ie the count of the freq of a word


# In[73]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=50000,stop_words='english')


# In[74]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[75]:


vectors


# In[76]:


cv.vocabulary_  #gives those 5k words with freq


# In[77]:


cv.get_feature_names()


# In[64]:


get_ipython().system('pip install nltk')


# In[65]:


import nltk


# In[70]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[71]:


def stem(text):
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[72]:


new_df['tags']=new_df['tags'].apply(lambda x: stem(x))


# In[81]:


from sklearn.metrics.pairwise import cosine_similarity


# In[82]:


similarity=cosine_similarity(vectors)


# In[83]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])


# ## Recommendation

# In[92]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
    


# In[93]:


recommend('Batman Begins')


# In[95]:


recommend('Spectre')


# In[97]:


recommend('John Carter')


# In[99]:


recommend('Octopussy')


# In[100]:


recommend('Avatar')

