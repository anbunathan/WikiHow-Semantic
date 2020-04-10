import pandas as pd
import os
import re
from nltk.tokenize import RegexpTokenizer
import spacy
EN = spacy.load('en')
from sklearn.model_selection import train_test_split
from pathlib import Path

# read data from the csv file (from the location it is stored)
df = pd.read_csv(r'wikihowAll-trial.csv', engine='python', encoding = "latin1")
df = df.head(1000)
df.drop(df.columns.difference(['headline','title','text']), 1, inplace=True)
print(df.head(5))
print(df.columns)

def tokenize(text):
    "A very basic procedure for tokenizing code strings."
    return RegexpTokenizer(r'\w+').tokenize(text)

# tokenizer = RegexpTokenizer("[\w']+")
# df['headline_tokens'] = df['headline'].apply(tokenizer.tokenize)
# print(df['headline_tokens'].head())

pd.options.display.max_colwidth = 100
pd.options.display.max_rows = 999
df.dropna()
df.apply(str).apply(str.lower)
df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=' ', regex=True, inplace=True)
df.replace(to_replace='[^\w\s]', value='', regex=True, inplace=True)
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
print(df['headline'].head(5))

# df["headline"] = df['headline'].str.replace('[^\w\s]','')
# df['headline'] = df['headline'].apply(lambda x: str(x).lower())
# nlp = spacy.load('en')
# df['headline_tokens'] = df['headline'].apply(lambda x: nlp(str(x)))
# print(df['headline_tokens'].values)


# # df = df.apply(lambda x: str(x).lower())
# nlp = spacy.load('en')
# df['headline_token'] = df['headline'].apply(lambda x: nlp(str(x)))
# print(df.head(5))

def tokenize (column):
    tokens = []
    nlp = spacy.load('en')
    for doc in nlp.pipe(column.astype('unicode').values, batch_size=50,
                        n_threads=3):
        if doc.is_parsed:
            tokens.append(' '.join([n.text.lower() for n in doc]))
        else:
            # We want to make sure that the lists of parsed results have the
            # same number of entries of the original Dataframe, so add some blanks in case the parse fails
            tokens.append(None)
    return tokens
df['headline_token'] = tokenize (df['headline'])
df['title_token'] = tokenize (df['title'])
df['text_token'] = tokenize (df['text'])
# print(df['headline_token'].head(5))
# print(df['title_token'].head(5))
# print(df['text_token'].head(5))
df.headline_token.to_csv('headline.csv')

grouped = df.groupby('headline')

train, test = train_test_split(df, train_size=0.87, shuffle=True, random_state=8081)
train, valid = train_test_split(train, train_size=0.82, random_state=8081)

print(train.head(5))

def write_to(df, filename, path='./data/processed_data/'):
    "Helper function to write processed files to disk."
    out = Path(path)
    out.mkdir(exist_ok=True)
    df.text.to_csv(out/'{}.function'.format(filename), index=False)
    df.headline.to_csv(out/'{}.docstring'.format(filename), index=False)
    df.title.to_csv(out/'{}.lineage'.format(filename), index=False)

# write to output files
write_to(train, 'train')
write_to(valid, 'valid')
write_to(test, 'test')



