#import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('Téléchargements'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from nltk import WordNetLemmatizer
#from nltk.corpus import stopwords
##from gensim.utils import lemmatize
from sklearn import model_selection, linear_model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas


# Telecharger la dataset 
train_path = 'C:\\Users\\hp\\Downloads\\train.csv'
test_path ='C:\\Users\\hp\\Downloads\\test.csv'
submission_path = 'C:\\Users\\hp\\Downloads\\submission_df2.csv'

# Lire la  dataset
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission_sample = pd.read_csv(submission_path)

#  5 prem lignes de train dataset
train_df.head(5)

# 5 prem lignes de test dataset
test_df.head(5)

# 5 prem lignes de submission dataset
submission_sample.head(5)

# Forme de la dataset
print("Total number of rows in train dataset are ",train_df.shape[0],'and total number of columns in train dataset are',train_df.shape[1])
print("Total number of rows in test dataset are ",test_df.shape[0],'and total number of columns in test dataset are',test_df.shape[1])

# Info basiques de train dataset
train_df.info()

#Info basiques de test data
test_df.info()

#Valeurs nulles ds train dataset
train_df.isnull().sum()
print(train_df.isnull().sum())

#Valeurs nulles ds test dataset
test_df.isnull().sum()
print(test_df.isnull().sum())

train_df.isna().sum().plot(kind="bar")
plt.title("nbr des valeurs nulles dans  train data")
plt.show()


test_df.isna().sum().plot(kind="bar")
plt.title("nbr des valeurs nulles dans test data")
plt.show()

# Suppression des colonnes Location et Keyword
train_df = train_df.drop(['location','keyword'],axis=1)
test_df = test_df.drop(['location','keyword'],axis=1)
# Entrainement apres suppression des colonnes
train_df.head()



# Trouver le pourcentage de 0 et 1 de target
real_tweets = len(train_df[train_df["target"] == 1])
real_tweets_percentage = real_tweets/train_df.shape[0]*100
fake_tweets_percentage = 100-real_tweets_percentage
print("Pourcentage des real Tweets : ",real_tweets_percentage)
print("Pourcentage des fake Tweets : ",fake_tweets_percentage)

# Tracer les valeurs de target
sns.countplot(x='target',data=train_df)
plt.show()

length_train = train_df['text'].str.len() 
length_test = test_df['text'].str.len() 
plt.hist(length_train, label="train_tweets") 
plt.hist(length_test, label="test_tweets") 
plt.legend() 
plt.show()

# disaster tweets
disaster_tweets = train_df[train_df['target'] ==1 ]['text']
for i in range(1,10):
    print(disaster_tweets[i])
    
# non-disaster tweets
non_disaster_tweets = train_df[train_df['target'] !=1 ]['text']

# word cloud de disaster et non-disaster tweets
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 5])
wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(disaster_tweets))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Disaster Tweets',fontsize=40);

wordcloud2 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(non_disaster_tweets))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Non Disaster Tweets',fontsize=40);

# Nettoyage du texte
def clean_text(text):
    ''' Rendre le texte en minuscule , supprimer : le texte entre [],liens , ponctuation , les textes contenant des nbrs  .'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Appliquer la fct de nettoyage pour  test et train datasets
train_df['text'] = train_df['text'].apply(lambda x: clean_text(x))
test_df['text'] = test_df['text'].apply(lambda x: clean_text(x))



# Mettre a jr le text
train_df['text'].head()

tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+')
train_df['text'] = train_df['text'].apply(lambda x:tokenizer.tokenize(x))
test_df['text'] = test_df['text'].apply(lambda x:tokenizer.tokenize(x))
train_df['text'].head()

# stopwords
stopwords.words('english')
print(stopwords.words('english'))
#nbr des  stopwords
len(stopwords.words('english'))
print(len(stopwords.words('english')))

# supression des stopwords
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

train_df['text'] = train_df['text'].apply(lambda x : remove_stopwords(x))
test_df['text'] = test_df['text'].apply(lambda x : remove_stopwords(x))
test_df.head()

# lemmatization
lem = WordNetLemmatizer()
def lem_word(x):
    return [lem.lemmatize(w) for w in x]

train_df['text'] = train_df['text'].apply(lem_word)
test_df['text'] = test_df['text'].apply(lem_word)
train_df['text'][:10]

def combine_text(list_of_text):
    '''Prendre des listes de texte et les combiner en un seul texte  .'''
    combined_text = ' '.join(list_of_text)
    return combined_text

train_df['text'] = train_df['text'].apply(lambda x : combine_text(x))
test_df['text'] = test_df['text'].apply(lambda x : combine_text(x))
train_df['text']
train_df.head()


##CountVectorizer
count_vectorizer = CountVectorizer()
train_vector = count_vectorizer.fit_transform(train_df['text'])
test_vector = count_vectorizer.transform(test_df['text'])
print(train_vector[0].todense())


tfidf = TfidfVectorizer(min_df = 2,max_df = 0.5,ngram_range = (1,2))
train_tfidf = tfidf.fit_transform(train_df['text'])
test_tfidf = tfidf.transform(test_df['text'])
tfidf.transform(test_df['text'])

##Logistic Regression Model
lg = LogisticRegression(C = 1.0)
scores_vector = model_selection.cross_val_score(lg, train_vector, train_df["target"], cv = 5, scoring = "f1")
print("Logistic Regression score:",scores_vector)
scores_tfidf = model_selection.cross_val_score(lg, train_tfidf, train_df["target"], cv = 5, scoring = "f1")
print("score of tfidf:",scores_tfidf)


## RidgeClassifier Model.
clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vector, train_df["target"], cv=5, scoring="f1")
scores
print("Ridge Classifier :" ,scores)

##XGBoost Algorithm
xgb_param = xgb.XGBClassifier(max_depth=5,n_estimators=500,colsample_bytree=0.8,nthread=10,learning_rate=0.05)
scores_vector = model_selection.cross_val_score(xgb_param,train_vector,train_df['target'],cv=5,scoring='f1')
scores_vector
print("XGBoost score :" , scores_vector)



##Prediction
lg.fit(train_tfidf, train_df["target"])
y_pred = lg.predict(test_tfidf)
print(y_pred)

##Submission
submission_df2 = pd.DataFrame({'Id':test_df['id'],'target':y_pred})
submission_df2.to_csv('submission_df2.csv',index=False)
submission_df2 = pd.read_csv('submission_df2.csv')
submission_df2.head()