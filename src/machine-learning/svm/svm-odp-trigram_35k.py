import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

random_state = 47
np.random.seed(seed=random_state)


df = pd.read_csv('./odp_35_amostras.csv')

df = df.dropna()

print('Data set len: ', len(df))

print(df.head())


dict_cat = {
    'Adult': 0,
    'Arts': 1,
    'Business': 2,
    'Computers': 3,
    'Games': 4,
    'Health': 5,
    'Home': 6,
    'Kids': 7,
    'Recreation': 8,
    'Reference': 9,
    'Science': 10,
    'Shopping': 11,
    'Society': 12,
    'Sports': 13
}

def to_category_id(item):
    return dict_cat[item]


df['cat_id'] = df['category'].apply(to_category_id)
print(df.head())


def length_char(item):
    return len(remove_special_char(item))


def remove_special_char(item):
    return re.sub(r'\W+', '', item)


regex_replace_http = r'(www[0-9][.])'
def replace_http(item):
    item_r = item.replace('http://', '').replace('https://', '').replace('www.', '')
    return re.sub(regex_replace_http, '', item_r)


def qt_number(item):
    return len(''.join(re.findall(r'[0-9]+', item)))


regex_split = r'[./=~?&+\'\"_;-]+(?=[\w])+'
def qt_tokens(item):
    return len(re.split(regex_split,item))


regex_split = r'[./=~?&+\'\"_;-]+(?=[\w])+'
def average_len_tokens(item):
    tokens = re.split(regex_split,item)
    return sum(len(remove_special_char(token)) for token in tokens) / len(tokens)


def get_hostname_len(item):
    return len(remove_special_char(item.split('/')[0]))

def only_char(item):
    return re.sub('[^A-Za-z]+', '', item)


df['n_url'] = df['url'].apply(replace_http)


df['norm_url'] = df['n_url'].apply(remove_special_char)
df['url_text'] = df['n_url'].apply(only_char)

print(df.head())

X = df['url_text']
Y = df['cat_id']


count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3)).fit(X)
print('Length of vocabulary: ', len(count_vectorizer.vocabulary_))

#words_vector = count_vectorizer.transform(X)
urls_tf = count_vectorizer.transform(X)

#tf_transformer = TfidfTransformer(norm=None,use_idf=False).fit(words_vector)
#urls_tf = tf_transformer.transform(words_vector)


print('TF shape: ', urls_tf.shape)


#data_set = df.ix[:,6:].as_matrix()
#print('New features shape: ', data_set.shape)
#print('Data set with new features: ', data_set)


#import scipy as sp

#urls_tf = sp.sparse.hstack((urls_tf, data_set[:,:]))


print('Data set shape: ', urls_tf.shape)
print('First item: ', urls_tf.getrow(0).toarray())


from sklearn.preprocessing import normalize


#print('Normalizing data...')
#urls_tf = normalize(urls_tf) 


print('Final data set: ', urls_tf.shape)
print('First element: ', urls_tf.getrow(0).toarray())

print('Labels shape: ', Y.shape)

#%%

url_train,url_test,label_train,label_test = train_test_split(urls_tf, Y, test_size=0.3,random_state=random_state)

X = None
Y = None
urls_tf = None
data_set = None
df = None

print('Train shape: ', url_train.shape)
print('Labels train shape: ',label_train.shape)

#url_train = url_train.todense()
#label_train = label_train

print('Data set train shape: ', url_train.shape)
print('Label train shape: ', label_train.shape)

#%%

print('\n\n================== Starting training ==================\n\n')

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import datetime

C = 5.5
gamma = 0.05
max_iter = 40000

print('C: ', C)
print('Gamma: ', gamma)
print('Max iter: ', max_iter)

print('Start: ', datetime.datetime.now())

model = SVC(C=C, gamma=gamma,random_state=random_state,max_iter=max_iter)
model.fit(url_train,label_train)

print('Finish: ', datetime.datetime.now())

print('Finish training')

predictions = model.predict(url_test)

print(classification_report(label_test, predictions))

np.save('./resultados/labels_35k_OnlyTrigram', np.array(label_test))
np.save('./resultados/saidas_35k_OnlyTrigram', np.array(predictions))
