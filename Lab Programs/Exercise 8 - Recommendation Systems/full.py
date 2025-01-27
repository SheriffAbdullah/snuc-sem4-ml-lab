#!/usr/bin/env python
# coding: utf-8

# <h4> Question: Use any recommendation dataset from Kaggle and perform content based and collobarative filtering on the dataset. </h4>

# Inspiration: https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
# 
# There's a lot of preprocessing & cool steps in this article, do check them out!
# 
# For Theory: https://towardsdatascience.com/a-complete-guide-to-recommender-system-tutorial-with-sklearn-surprise-keras-recommender-5e52e8ceace1

# <h3> Content-Based Recommendation System </h3>

# In[1]:

import pandas as pd
import numpy as np
import random
import math

import warnings
warnings.filterwarnings('ignore')

# In[2]:

data = pd.read_csv("Netflix_Dataset_Movie.csv")

print(data.head())
print()


# In[3]:

def WordCleaner(word):
    word = word.lower()
    letters = list(word)
    
    word = ""
    
    for letter in letters:
        if letter.isalpha() or letter.isnumeric():
            word += letter
    
    return word


# In[4]:

def SentenceCleaner(sentence):
    cleaned_sentence = ""
    
    for word in sentence.split():
        word = WordCleaner(word)
        cleaned_sentence += word + " "
    
    # 'rstrip()' -> To remove the extra ' ' after the last word
    return cleaned_sentence.rstrip()


# In[5]:

# Example use of WordCleaner()

WordCleaner("What's")


# In[6]:

# Example use of Sentence Cleaner()

SentenceCleaner("Isle of Man TT 2004 Review")


# In[7]:

# Create a new 'Cleaned Name' column

for i in range(len(data)):
    data.loc[i, 'Cleaned Name'] = SentenceCleaner(data.loc[i, 'Name'])
    
print(data.head(10))
print()


# In[8]:

# data = data.drop("Name", axis=1)


# In[9]:

# Create a list of unique words in entire dataset


unique_words = set() # Using sets, since set does not store duplicate entries.

# For every movie name
for name in data['Cleaned Name']:
    words = name.split()
    
    # Add every word in the name to the 'unique_words' set
    for word in words:
        unique_words.add(word)

# Convert to list and sort in alphabetical order
unique_words = list(unique_words)
unique_words.sort()

print("Last 10 Unique Words:")
print(unique_words[-10:])
print()


# In[10]:

def TermFrequencies(term, documents):
    '''
    tf(term, document) = count of 'term' in 'document' / number of terms in document
    '''
    
    tmp = []
    
    for document in documents:
        document = document.split()
        tmp.append(document.count(term) / len(document))

    return tmp


def DocumentFrequency(term, documents):
    '''
    df(term) = Occurrence of 'term' in 'N' documents
    '''
    
    count = 0
    
    for document in documents:
        if term in document.split():
            count += 1
            
    return count


def InverseDocumentFrequency(term, documents):
    '''
    idf(term) = log('N' Documents / (df(term) + 1))
    
    Note:
    '1' is to smoothen the value, if the word is absent in the document, since 'df(term)' will be '0'.
    'log' is to dampen the effect of a large corpus (say N >= 10000)
    '''
    
    return math.log(len(documents) / (DocumentFrequency(term, documents) + 1))


def TFIDF(term, documents):
    '''
    tfidf(term, document) = tf(term, document) * idf(term)
    '''
    
    tf = TermFrequencies(term, documents)
    idf = InverseDocumentFrequency(term, documents)
    
    tmp = []
    for idx, document in enumerate(documents):
        tmp.append(tf[idx] * idf)
        
    return tmp


# In[11]:

# Cosine Similarity

# cosine(x, y) = x.y / ||x||.||y||
# where ||x|| = sqrt(x1^2 + x2^2 + ... + xn^2)

def CosineSimilarity(x, y):
    tmp = np.dot(x, y)
    
    x_norm = 0
    y_norm = 0
    
    for i in range(len(x)):
        x_norm += x[i] ** 2
        y_norm += x[i] ** 2
        
    x_norm = math.sqrt(x_norm)
    y_norm = math.sqrt(y_norm)
    
    return tmp / (x_norm * y_norm)


# In[12]:

# Calculate TFIDF for every movie name, and make a TFIDF table
tfidf = []

for word in unique_words:
    tfidf.append(TFIDF(word, data['Cleaned Name']))


# In[13]:

# From (num of unique words x num of documents) -> To (num of documents x num of unique words)

tfidf = np.array(tfidf).T


# In[14]:

print("Shape of TFIDF Matrix:")
print(tfidf.shape)
print()


# In[15]:

# This is going to take a LOT of time
# Since it has 17770 x 17770 = 31 Crore Cosine Operations to perform
# At 1 second per operation, this might take 3600+ days to run.

'''
scores = []

for i in range(len(data)):
    tmp = []
    
    for j in range(len(data)):
        tmp.append(CosineSimilarity(tfidf[i], tfidf[j]))
    
    if i % 1000 == 0:
        print(i, "documents done.")
    
    scores.append(tmp)
'''


# In[16]:

movie_preference = input("Enter a keyword of your favourite movie: ").lower()

# Print the movie names matching the 'movie_preference'
movie_found = False
for idx, movie_name in enumerate(data['Cleaned Name']):
    if movie_preference in movie_name:
        movie_found = True
        print(f"ID: {idx} \t | {data['Name'][idx]}")

if not movie_found:
    print("Movie not found in database, please enter another movie name.")
    
# Get the movie index, so that we can search for similar movies
user_preference_index = int(input("Enter the ID of the movie: "))


# In[17]:

def RecommendMovies(movie_index, tfidf, recomm_count=5):
    score = []
    
    # For every movie in the database
    print("\nFinding similar movies.")
    for i in range(len(tfidf)):
        # We don't want to recommend the SAME movie as input
        if i == movie_index: # If the movie to be compared is the input, omit it.
            score.append(0)
        else:
            # Find the cosine similarity between the movies
            score.append(CosineSimilarity(tfidf[movie_index], tfidf[i]))
        
        if i % 1000 == 0:
            print(i, "movies compared.")
    
    # Sort based on movie similarity
    rank = sorted(enumerate(score), key=lambda x: x[1], reverse=True)
    
    # Using the ranked movie indices, get the name of the movie, and return the recommendations
    recommended_movies = []
    for i in range(recomm_count):
        recommended_movies.append(data['Name'][rank[i][0]] + " (" + str(data['Year'][rank[i][0]]) + ")")
        
    return recommended_movies


# In[18]:

# Uses cosine similarity score, and gives top 'n' movies
print("\n*** Recommended Movies using TFIDF & Cosine Similarity ***")
print(RecommendMovies(user_preference_index, tfidf))
print()


# In[19]:

# Experimental CodeBlock:

# Try Recommendation With ONLY TF-IDF,
# using the keyword (movie preference) itself, and not the movie index.

names = sorted(enumerate(TFIDF(movie_preference, data['Cleaned Name'])), reverse=True, key=lambda x: x[1])[:5]

num_movies = 5

print("*** Recommended Movies using TFIDF only ***")
print(f"Top {num_movies} movies similar to '{movie_preference}' are:\n")

for i in names:
    print(data['Name'][i[0]] + " (" + str(data['Year'][i[0]]) + ")")
    
print()

# In this method, the distance between movie vectors is considered.
# In the previous method, the angle between the 2 vectors were taken into consideration


# <h3> Using Scikit-Learn </h3>

# Source: https://medium.com/@sumanadhikari/building-a-movie-recommendation-engine-using-scikit-learn-8dbb11c5aa4b

# In[20]:

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

count_matrix = cv.fit_transform(data['Cleaned Name'])


# In[21]:

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()

tfidf_matrix = tf.fit_transform(data['Cleaned Name'])


# In[22]:

print("Selected Features (or) Unique Words by 'scikit' library.")
print(cv.get_feature_names_out())
print()

count_matrix.toarray()


# In[23]:

# Calculate Cosine Similarity

from sklearn.metrics.pairwise import cosine_similarity

similarity_scores = cosine_similarity(tfidf_matrix)
similarity_scores[:5]


# In[24]:

# Let's print the similarity scores for the first word
# for idx, i in enumerate(similarity_scores[0]):
    # if i > 0:
        # print(idx, i)


# In[25]:

data['Cleaned Name'][0]


# In[26]:

data['Cleaned Name'][4099]
data['Cleaned Name'][6046]


# In[27]:

movie_preference = input("\nEnter a keyword of your favourite movie: ")


# In[28]:

movie_found = False

for idx, movie_name in enumerate(data['Cleaned Name']):
    if movie_preference.lower() in movie_name:
        movie_found = True
        print(f"ID: {idx} \t | {data['Name'][idx]}")
        
if not movie_found:
    print("Movie not found in database, please enter another movie name.")


# In[29]:

user_preference_index = int(input("Enter the ID of the movie: "))


# In[30]:

similar_movies = list(enumerate(similarity_scores[user_preference_index]))
sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)[1:]


# In[31]:

sorted_similar_movies


# In[32]:

i = 0

print("\n*** Recommended Movies using Scikit-learn library ***")
print(f"Top 5 movies similar to '{data['Name'][user_preference_index]}' using 'scikit' are:\n")

for elt in sorted_similar_movies:
    print(f"#{i+1} {data['Name'][elt[0]]} ({data['Year'][elt[0]]})")
    i += 1
    
    if i == 5: break


# <h2>Collaborative Filtering based Recommendation</h2>

# Inspiration: https://www.youtube.com/watch?v=3oCtj29XeYY

# In[33]:

rows = len(data)
rows


# In[34]:

def AddUser(pref, non_pref):
    pref_movies = []
    non_pref_movies = []

    # Get list of preferred movies & non-preferred movies
    for i in range(rows):
        if pref in data['Cleaned Name'][i]:
            pref_movies.append(i)
        if non_pref in data['Cleaned Name'][i]:
            non_pref_movies.append(i)
            
    user = []
    
    good_ratings = [8, 9, 10]
    bad_ratings = [2, 3, 4]
    watched = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1] # Giving more '1's, so there is a '1' in '10' probability movie is NOT watched.
    
    for i in range(rows):               
        if i in pref_movies:
            # Give a random chance that they watched a movie
            # They need not have watched every movie in given genre
            if random.choice(watched):
                user.append(random.choice(good_ratings))
            else:
                user.append(np.nan)
        elif i in non_pref_movies:
            if random.choice(watched):
                user.append(random.choice(bad_ratings))
            else:
                user.append(np.nan)
        else: # User did not interact with movie
            user.append(np.nan)

    user = pd.DataFrame(user).T
    user.columns = data['Cleaned Name']
    return engagement.append(user)


# In[35]:

# Let's create an Engagement Table
engagement = pd.DataFrame(columns=data['Cleaned Name'])

# Let's create a User's profile who likes Dogs, and doesn't like Dance movies
engagement = AddUser(pref='dog', non_pref='dance')

# Let's create a User's profile who likes fighting, and doesn't like Romancee
engagement = AddUser(pref='fight', non_pref='romance')

# Let's create a User's profile who likes violent movies, and doesn't like Child movies
engagement = AddUser(pref='kill', non_pref='child')

# Let's create a User's profile who likes Dog movies, and doesn't like Child movies
engagement = AddUser(pref='dog', non_pref='child')

engagement = engagement.reset_index(drop=True)


# In[36]:

# To normalize the score from '0' to '10' -> To make it [-5, 5]
engagement = engagement.apply(lambda x: x - 5)


# In[37]:

engagement.head(10)


# In[38]:

# Consider '0' score as 'NOT Interacted'
for i in range(len(engagement)):
    for j in range(len(engagement.T)):
        if np.isnan(engagement.iloc[i, j]):
            engagement.iloc[i, j] = 0


# In[39]:

engagement = engagement.T


# In[40]:

print("\n*** Utility Matrix ***")
print(engagement.head())


# In[41]:

# To see the movie names with 'dog' in it

dog_movies = []
for i in data['Cleaned Name']:
    if 'dog' in i:
        dog_movies.append(i)
        
dog_movies[:15]


# In[42]:

# Now, let's find the prediction for User 4
new_user = 3

# Find similarity of a user with others
similarity_matrix = []

for i in range(len(engagement.T)):
    # Obviously, User 4 is MOST similar to User 4, so make the score '0' intentionally
    if i == new_user:
        similarity_matrix.append(0)
    else:
        similarity_matrix.append(CosineSimilarity(engagement[new_user], engagement[i]))


# In[43]:

similarity_matrix


# In[44]:

sorted(enumerate(similarity_matrix), key=lambda x: x[1], reverse=True)


# In[45]:

most_similar_user = sorted(enumerate(similarity_matrix), key=lambda x: x[1], reverse=True)[0][0]


# In[46]:

print(f"\nMost similar user is: User {most_similar_user + 1}")


# In[47]:

engagement


# In[48]:

n_movies = 5

print(f"\nTop {n_movies} recommended movies based on similar Users are: \n")

i = 0
for movie in sorted(enumerate(list(engagement.iloc[:, most_similar_user])), key=lambda x: x[1], reverse=True):
    if engagement[new_user][movie[0]] == 0:
        print(f"#{i+1} {data['Name'][movie[0]]} ({data['Year'][movie[0]]})")
        i += 1
        
    if i >= n_movies: break


# In[49]:

# To Experiment:
# 1. With more data pre-processing (i.e.) Stemming, Lemmatization, & more
# 2. Hybrid Recommendation System = Collaborative-filtering + Content-based
# 3. Put code it into more functions, for ease-of-use


# In[ ]:




