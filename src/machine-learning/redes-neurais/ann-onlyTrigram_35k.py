import pandas as pd
import numpy as np
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

random_state = 47
np.random.seed(seed=random_state)

df = pd.read_csv('./../../data-sets/odp_35_amostras.csv')

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

def remove_special_char(item):
    return re.sub(r'\W+', '', item)


regex_replace_http = r'(www[0-9][.])'
def replace_http(item):
    item_r = item.replace('http://', '').replace('https://', '').replace('www.', '')
    return re.sub(regex_replace_http, '', item_r)


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

urls_tf = count_vectorizer.transform(X)
print('TF shape: ', urls_tf.shape)


data_set = df.ix[:,6:].as_matrix()
print('New features shape: ', data_set.shape)
print('Data set with new features: ', data_set)
print('Data set shape: ', urls_tf.shape)
print('First item: ', urls_tf.getrow(0).toarray())
print('Labels shape: ', Y.shape)


#%%

url_train,url_test,label_train,label_test = train_test_split(urls_tf, Y, test_size=0.3)

X = None
Y = None
urls_tf = None
data_set = None
df = None

url_train = url_train.todense()
n_feat = url_train.shape[1]

print('Teste shape: ', url_test.shape)
print('Labels teste shape: ',label_test.shape)

url_test = url_test.todense()
label_test = label_test

print('Data set teste shape: ', url_test.shape)
print('Label teste shape: ', url_test.shape)

#%%

print('\n\n================== Starting testing ==================\n\n')


import tflearn
from tflearn.data_utils import to_categorical

#%%
n_epoch = 50
classes = 14
hidden_layer_size = int((n_feat * 2)/ 3 + classes)
print('hidden layer size: ', hidden_layer_size)
label_test = to_categorical(label_test, nb_classes=classes)
#%%

# Building deep neural network
input_layer = tflearn.input_data(shape=[None, n_feat])
dense1 = tflearn.fully_connected(input_layer, hidden_layer_size, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout1 = tflearn.dropout(dense1, 0.5)
dense2 = tflearn.fully_connected(dropout1, hidden_layer_size//2, activation='tanh',
                                 regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(dense2, 0.5)
softmax = tflearn.fully_connected(dropout2, classes, activation='softmax')

sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,
                         loss='categorical_crossentropy')

## Treinamento
model = tflearn.DNN(net, best_checkpoint_path='./models/model_30k_OnlyTri',
    max_checkpoints=3,best_val_accuracy=0.3)

#%% FIT E SALVA
model.fit(url_train, label_train, n_epoch=n_epoch, show_metric=True,
          run_id='model_30k_OnlyTri', snapshot_epoch=True,validation_set=0.2)

