#!/usr/bin/env python3

import re
import warnings

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')


def text_cleaning(text):
    '''
    Cleans text into a basic form for NLP. Operations include the following:-
    1. Remove special charecters like &, #, etc
    2. Removes extra spaces
    3. Removes embedded URL links
    4. Removes HTML tags
    5. Removes emojis

    text - Text piece to be cleaned.
    '''

    template = re.compile(r'https?://\S+|www\.\S+')  # Removes website links
    text = template.sub(r'', text)

    soup = BeautifulSoup(text, 'lxml')  # Removes HTML tags
    only_text = soup.get_text()
    text = only_text

    emoji_pattern = re.compile('['
                               u'\U0001F600-\U0001F64F'  # emoticons
                               u'\U0001F300-\U0001F5FF'  # symbols & pictographs
                               u'\U0001F680-\U0001F6FF'  # transport & map symbols
                               u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                               u'\U00002702-\U000027B0'
                               u'\U000024C2-\U0001F251'
                               ']+', flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = re.sub(r'[^a-zA-Z\d]', ' ', text)  # Remove special Charecters
    text = re.sub(' +', ' ', text)  # Remove Extra Spaces
    text = text.strip()  # Remove spaces at the beginning and at the end of string

    return text


def train(data_path):
    cat_weights = {'obscene': 0.16,
                   'toxic': 0.32,
                   'threat': 1.5,
                   'insult': 0.64,
                   'severe_toxic': 1.5,
                   'identity_hate': 1.5}

    df_train = pd.read_csv(data_path)
    df_train['y'] = sum([df_train[c] * w for c, w in cat_weights.items()])
    df_train['comment_text'] = df_train['comment_text'].progress_apply(text_cleaning)

    n_toxic = (df_train['y'] >= 0.1).sum()
    df_train_balanced = pd.concat([df_train[df_train['y'] >= 0.1],
                                   df_train[df_train['y'] == 0].sample(n=n_toxic, random_state=0)])

    vectorizer = TfidfVectorizer(min_df=3, max_df=0.5, analyzer='char_wb', ngram_range=(3, 5))
    x = vectorizer.fit_transform(df_train_balanced['comment_text'])
    y = df_train_balanced['y']

    model = Ridge(alpha=0.5)
    model.fit(x, y)

    return vectorizer, model


def test(data_path, vectorizer, model):
    df_test = pd.read_csv(data_path)
    df_test['less_toxic'] = df_test['less_toxic'].progress_apply(text_cleaning)
    df_test['more_toxic'] = df_test['more_toxic'].progress_apply(text_cleaning)

    x_less_toxic = vectorizer.transform(df_test['less_toxic'])
    x_more_toxic = vectorizer.transform(df_test['more_toxic'])

    p1 = model.predict(x_less_toxic)
    p2 = model.predict(x_more_toxic)

    acc = (p1 < p2).mean()

    print(f'Test accuracy: {acc}')


def main():
    train_data_path = '/media/golovanov/74b0c986-d950-458d-907c-8205cf9a8bbe/kaggle/jigsaw/train.csv'
    test_data_path = '/media/golovanov/74b0c986-d950-458d-907c-8205cf9a8bbe/kaggle/jigsaw/validation_data.csv'

    tqdm.pandas()
    vectorizer, model = train(train_data_path)
    test(test_data_path, vectorizer, model)


if __name__ == '__main__':
    main()
