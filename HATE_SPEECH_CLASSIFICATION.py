# import all packages 
import pymongo
from pymongo import MongoClient
import json
import twitter
#from pprint import pprint
import pandas as pd
import numpy as np
import gc
#import sys
#import itertools
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
#import string
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import *
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from random import seed
from random import randrange
from csv import reader
import emoji
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import os

########## DATA PROCESSING :: START ####################################
# code to clean text once tweets are collected from API
def strip_non_ascii(string):
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)

def clean_texts(tweet):
    if tweet:
        tweet = strip_non_ascii(tweet)
        tweet= BeautifulSoup(tweet, "lxml")
        tweet= tweet.get_text()
        tweet= tweet.replace('\n', '').replace('\r', ' ').replace('\t', ' ')
        tweet= tweet.replace('!', '')###### remove !
        tweet= tweet.replace('"', '')###### remove "
        tweet = re.sub(r'^http?:\/\/.*[\r\n]*', '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r"http\S+", "", tweet, flags=re.MULTILINE)
        # remove between word dashes
        tweet= tweet.replace('- ', '').replace(' -','').replace('-','')
        #replace parentheses
        tweet= tweet.replace("(","").replace(")","").replace("[","").replace("]","").replace("RT","")
        #remove punctuation but keep commas, semicolons, periods, exclamation marks, question marks, intra-word dashes and apostrophes (e.g., "I'd like")
        tweet= tweet.replace(r"[^[:alnum:][:space:].'-:]", "").replace('+','').replace('*','').replace("' ","").replace(" '","").replace("'","").replace(","," ").replace(";"," ").replace(":"," ").replace("."," ")
        #remove numbers (integers and floats)
        tweet= re.sub('\d+', '', tweet)        
        #remove extra white space, trim and lower
        tweet = re.sub('\\s+',' ',tweet).strip()
        return tweet
    else:
        return ""    

def extract_emojis(tweet):
  return ''.join(c for c in tweet if c in emoji.UNICODE_EMOJI)

def remove_emoji(tweet):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', tweet)


# ## REPLACE A LL URLS, WHITE SPACES AND MENTIONS 
def preprocess(tweet):
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', tweet)
    parsed_text = re.sub(giant_url_regex, ' ', parsed_text)
    parsed_text = re.sub(mention_regex, ' ', parsed_text)
    # parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text


# Text Lemmatization
lemma = WordNetLemmatizer()
def lemmatize_words(doc):
    x = " ".join(lemma.lemmatize(word) for word in doc.split())
    return x

# Identifying Stop Words
stopwords_set = set(stopwords.words('english'))
def remove_stop_words(doc):
    x = " ".join([word for word in doc.lower().split() if word not in stopwords_set])
    return x

# Populate labelled dataset
HateSpeech_Offensive_Dataset = pd.read_csv('D:/NCI/4.HATE_SPEECH/SUBMISSION/FINAL_CODE/HateSpeech_Offensive_Dataset.csv')
HateSpeech_Offensive_Dataset.rename(columns = {'class' : 'label'}, inplace = True)

# ## remove emojis and clean text. Step one in preprocessing data 
HateSpeech_Offensive_Dataset['tweet'] = HateSpeech_Offensive_Dataset['tweet'].map(clean_texts)
HateSpeech_Offensive_Dataset['tweet'] = HateSpeech_Offensive_Dataset['tweet'].map(preprocess)
HateSpeech_Offensive_Dataset['tweet_emojis'] = HateSpeech_Offensive_Dataset['tweet'].map(extract_emojis)
HateSpeech_Offensive_Dataset['tweet'] = HateSpeech_Offensive_Dataset['tweet'].map(remove_emoji)
HateSpeech_Offensive_Dataset['tweet'] = HateSpeech_Offensive_Dataset['tweet'].map(remove_stop_words)
HateSpeech_Offensive_Dataset['tweet'] = HateSpeech_Offensive_Dataset['tweet'].map(lemmatize_words)

HateSpeech_Offensive_Dataset.head()

########## DATA PROCESSING :: END ####################################

########## FEATURE EXTRACTION :: START ##############################
"""
Features Used:
1. Count Vectorizer: X_train_count, X_test_count
2. Uni-Gram TF-IDF: X_train_tfidf, X_test_tfidf
3. N-Grams TF-IDF: X_train_tfidf_ngram, X_test_tfidf_ngram
"""
# split dataset into train and test
from sklearn.cross_validation import train_test_split

X_train_text, X_test_text, y_train_label, y_test_label = train_test_split(HateSpeech_Offensive_Dataset['tweet'], 
                                                                          HateSpeech_Offensive_Dataset['label'], 
                                                                          test_size=0.2)

# Count Vector is a matrix notation of the dataset. It converts text documents into a matrix of token counts. It produces a sparse representation of term counts.
from sklearn.feature_extraction.text import CountVectorizer

# count vectorizer object 
vector_count = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
vector_count.fit(HateSpeech_Offensive_Dataset['tweet'])

# Convert training and test data using count vectorizer
X_train_count =  vector_count.transform(X_train_text)
X_test_count =  vector_count.transform(X_test_text)

# Tf-Idf vector will be developed where every element represents the tf-idf score of each term. 

from sklearn.feature_extraction.text import TfidfVectorizer

vector_tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words='english')
vector_tfidf.fit(HateSpeech_Offensive_Dataset['tweet'])

X_train_tfidf =  vector_tfidf.transform(X_train_text)
X_test_tfidf =  vector_tfidf.transform(X_test_text)

# #### N-gram level Tf-Idf

vector_ngram_tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3), stop_words='english')
vector_ngram_tfidf.fit(HateSpeech_Offensive_Dataset['tweet'])

X_train_tfidf_ngram =  vector_ngram_tfidf.transform(X_train_text)
X_test_tfidf_ngram =  vector_ngram_tfidf.transform(X_test_text)

# Funtion to plot Confusion matrix
def plot_conf_matrix (conf_matrix):
    class_names = [0,1,2]
    fontsize=14
    df_conf_matrix = pd.DataFrame(
            conf_matrix, index=class_names, columns=class_names, 
        )
    fig = plt.figure()
    heatmap = sns.heatmap(df_conf_matrix, annot=True, fmt="d")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

########## DATA MODELLING :: START ####################################
# ## Start :: Classification using Random Forest 

methods=[]
accuracy_scores=[]

# #### On word level Tf-Idf vector

from sklearn.ensemble import RandomForestClassifier

clf_rf_u = RandomForestClassifier(n_estimators=900, n_jobs=-1, verbose=1, oob_score = True)
clf_rf_u.fit(X_train_tfidf, y_train_label)
pred_rf_u = clf_rf_u.predict(X_test_tfidf)

from sklearn.metrics import accuracy_score

acc_score_rf_u = accuracy_score(y_test_label, pred_rf_u)
acc_score_rf_u

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import cohen_kappa_score

# ## Precision, Recall and F-Score measurement
pr_rf_u, re_rf_u, fs_rf_u, _ = precision_recall_fscore_support(y_test_label, pred_rf_u, average='macro')
kappa_rf_u = cohen_kappa_score(y_test_label, pred_rf_u)

methods.append('RF_unigram')
accuracy_scores.append(acc_score_rf_u)

from sklearn.metrics import confusion_matrix

conf_matrix_rf_u = confusion_matrix(y_test_label, pred_rf_u, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_rf_u)

# #### On N-gram Tf-Idf vector

clf_rf_n = RandomForestClassifier(n_estimators=900, n_jobs=-1, verbose=1, oob_score = True)
clf_rf_n.fit(X_train_tfidf_ngram, y_train_label)
pred_rf_n = clf_rf_n.predict(X_test_tfidf_ngram)

acc_score_rf_n = accuracy_score(y_test_label, pred_rf_n)
acc_score_rf_n

methods.append('RF_bigram')
accuracy_scores.append(acc_score_rf_n)

# ## Precision, Recall and F-Score measurement
pr_rf_n, re_rf_n, fs_rf_n, _ = precision_recall_fscore_support(y_test_label, pred_rf_n, average='macro')
kappa_rf_n = cohen_kappa_score(y_test_label, pred_rf_n)

conf_matrix_rf_n = confusion_matrix(y_test_label, pred_rf_n, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_rf_n)

# #### On Count Vector
clf_rf_c = RandomForestClassifier(n_estimators=900, n_jobs=-1, verbose=1, oob_score = True)
clf_rf_c.fit(X_traain_count, y_train_label)
pred_rf_c = clf_rf_c.predict(X_test_count)

acc_score_rf_c = accuracy_score(y_test_label, pred_rf_c)
acc_score_rf_c

methods.append('RF_c')
accuracy_scores.append(acc_score_rf_c)

conf_matrix_rf_c = confusion_matrix(y_test_label, pred_rf_c, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_rf_c)

# ## Precision, Recall and F-Score measurement
pr_rf_c, re_rf_c, fs_rf_c, last = precision_recall_fscore_support(y_test_label, pred_rf_c, average='macro')
kappa_rf_c = cohen_kappa_score(y_test_label, pred_rf_c)

# ## End :: Classification using Random Forest

# ## Start :: Support Vector Machine
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import *
from sklearn.metrics import cohen_kappa_score

clf_svm_c = svm.SVC(decision_function_shape='ovo', kernel = 'linear')
clf_svm_c.fit(X_train_count, y_train_label)
pred_svm_c = clf_svm_c.predict(X_test_count)

acc_score_svm_c = accuracy_score(y_test_label, pred_svm_c)

methods.append('SVM')
accuracy_scores.append(acc_score_svm_c)

conf_matrix_svm_c = confusion_matrix(y_test_label, pred_svm_c, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_svm_c)

# ## Precision, recall, FScore and Kappa
pr_svm_c, re_svm_c, fs_svm_c, last = precision_recall_fscore_support(y_test_label, pred_svm_c, average='macro')
kappa_svm_c = cohen_kappa_score(y_test_label, pred_svm_c)

# ## With decomposition
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=1000)
svd.fit(X_train_count)
X_train_count_decomp = svd.transform(X_train_count)

# calculate variance
svd.explained_variance_ratio_.sum()
X_train_count_decomp.shape

# transform test with decomposition function
X_test_count_decomp = svd.transform(X_test_count)
X_test_count_decomp.shape

clf_svm_decomp = SVC(decision_function_shape='ovo', kernel='linear')
clf_svm_decomp.fit(X_train_count_decomp, y_train_label)
pred_svm_decomp = clf_svm_decomp.predict(X_test_count_decomp)

acc_score_svm_decomp = accuracy_score(y_test_label, pred_svm_decomp)

methods.append('SVM_decomp')
accuracy_scores.append(acc_score_svm_decomp)

# ## Precision, recall, FScore and kappa
pr_svm_decomp, re_svm_decomp, fs_svm_decomp, _ = precision_recall_fscore_support(y_test_label, pred_svm_decomp, average='macro')
kappa_svm_decomp = cohen_kappa_score(y_test_label, pred_svm_decomp)

# Build confusion matrix
conf_matrix_svm_decomp = confusion_matrix(y_test_label, pred_svm_decomp, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_svm_decomp)

# #### On Uni-gram Tf-Idf vector
clf_svm_u = svm.SVC(decision_function_shape='ovo', kernel='linear')
clf_svm_u.fit(X_train_tfidf, y_train_label)
pred_svm_u = clf_svm_u.predict(X_test_tfidf)

acc_score_svm_u = accuracy_score(y_test_label, pred_svm_u)

methods.append('SVM_u')
accuracy_scores.append(acc_score_svm_u)

# ## Precision, recall, FScore and kappa
pr_svm_u, re_svm_u, fs_svm_u, _ = precision_recall_fscore_support(y_test_label, pred_svm_u, average='macro')
kappa_svm_u = cohen_kappa_score(y_test_label, pred_svm_u)

# plot confusion matrix
conf_matrix_svm_u = confusion_matrix(y_test_label, pred_svm_u, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_svm_u)

# #### On N-gram Tf-Idf vector
clf_svm_n = svm.SVC(gamma=1e-4,kernel='linear')
clf_svm_n.fit(X_train_tfidf_ngram, y_train_label)
pred_svm_n = clf_svm_n.predict(X_test_tfidf_ngram)

acc_score_svm_n = accuracy_score(y_test_label, pred_svm_n)

methods.append('SVM_n')
accuracy_scores.append(acc_score_svm_n)

# Precision, recall, Fscore and kappa
pr_svm_n, re_svm_n, fs_svm_n, _ = precision_recall_fscore_support(y_test_label, pred_svm_n, average='macro')
kappa_svm_n = cohen_kappa_score(y_test_label, pred_svm_n)

# plot confusion matrix
conf_matrix_svm_n = confusion_matrix(y_test_label, pred_svm_n, labels=None, sample_weight=None)
plot_conf_matrix(conf_matrix_svm_n)

# ## End :: Support Vector Machine

# ## Neural Networks
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.layers import * #LSTM, Activation, Dense, Dropout, Input, Embedding, SpatialDropout1D, MaxPooling1D, Convolution1D, GlobalMaxPool1D, SpatialDropout2D, Convolution2D, GlobalMaxPool2D, Conv1D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing import sequence

max_words = 1000
max_len = 450
lstm_out = 200
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train_text)
sequences = tok.texts_to_sequences(X_train_text)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
test_sequences = tok.texts_to_sequences(X_test_text)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

# ## Convolution Neural Network
def create_CNN_model():
    max_words = len(X_train_text)
    input_layer = Input(name='inputs',shape=[max_len])
    embedding_layer = Embedding(max_words, 450, weights=[sequences_matrix], trainable=True)(input_layer)
    embedding_layer = SpatialDropout1D(0.3)(embedding_layer)
    conv_layer = Conv1D(1000, 3, activation="relu")(embedding_layer)
    pooling_layer = GlobalMaxPool1D()(conv_layer)
    output_layer1 = Dense(500, activation="relu")(pooling_layer)
    output_layer1 = Dropout(0.25)(output_layer1)
    output_layer2 = Dense(3, activation="softmax")(output_layer1)
    model = Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn_model = create_CNN_model()
cnn_fit = cnn_model.fit(sequences_matrix, y_train_label, batch_size=128, epochs=10, validation_split=0.2)
cnn_predict = cnn_model.predict(test_sequences_matrix)
accr_cnn = cnn_model.evaluate(test_sequences_matrix,y_test_label)

methods.append('CNN')
accuracy_scores.append(accr_cnn)

# ## CNN with LSTM
def create_conv_lstm():
    model_conv = Sequential()
    model_conv.add(Embedding(max_words, 450, input_length=max_len))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(1000, 4, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(Dense(500, activation="relu"))
    model_conv.add(Dropout(0.25))
    model_conv.add(Dense(1, activation="softmax"))
    model_conv.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
    model_conv.add(Dense(3, activation='sigmoid'))
    model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv

lstm_conv_model = create_conv_lstm()
lstm_conv_fit = lstm_conv_model.fit(sequences_matrix, y_train_label, epochs = 10, validation_split=0.2)
cnn_lstm_predict = lstm_conv_model.predict(X_test_text)
accr_cnn_lstm = lstm_conv_model.evaluate(test_sequences_matrix,y_test_label)

methods.append('CNN_LSTM')
accuracy_scores.append(accr_cnn_lstm)

# ## RNN with LSTM
def create_RNN_model():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,450,input_length=max_len)(inputs)
    layer = LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2)(layer)
    layer = Dense(3,activation = 'softmax')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

rnn_model = create_RNN_model()
rnn_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
rnn_fit = rnn_model.fit(sequences_matrix, y_train_label, batch_size=128, epochs=10, validation_split=0.2)
rnn_predict = rnn_model.predict(X_test_text)
acc_score_rnn = rnn_model.evaluate(test_sequences_matrix,y_test_label)

methods.append('RNN_LSTM')
accuracy_scores.append(acc_score_rnn)

# END OF FINAL CODE ###################################### 
