import re
import spacy
import collections
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer

stop_words = stopwords.words('english')
nlp = spacy.load('en_core_web_lg')


# (NNPS|NNP|NNS|NN)
# (NNPS|NNP|NNS|NN) (NNPS|NNP|NNS|NN)
# (JJS|JJR|JJ) (NNPS|NNP|NNS|NN) (NNPS|NNP|NNS|NN)
# (JJS|JJR|JJ) (NNPS|NNP|NNS|NN)
# (RBR|RBS|RB) (JJS|JJR|JJ)
# (RBR|RBS|RB) (JJS|JJR|JJ) (NNPS|NNP|NNS|NN)
# (RBR|RBS|RB) (VBD|VBG|VBN|VBP|VBZ|VB)
# (RBR|RBS|RB) (RBR|RBS|RB) (JJS|JJR|JJ)
# (VBD|VBG|VBN|VBP|VBZ|VB) (NNPS|NNP|NNS|NN)
# (VBD|VBG|VBN|VBP|VBZ|VB) (RBR|RBS|RB)


patterns = "([a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(JJS|JJR|JJ) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN)|[a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(JJS|JJR|JJ) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN)|[a-zA-Z0-9'-]*_(JJS|JJR|JJ) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN)|[a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(JJS|JJR|JJ) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN)|[a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(JJS|JJR|JJ)|[a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(JJS|JJR|JJ)|[a-zA-Z0-9'-]*_(JJS|JJR|JJ) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN)|[a-zA-Z0-9'-]*_(RBR|RBS|RB) [a-zA-Z0-9'-]*_(VBD|VBG|VBN|VBP|VBZ|VB)|[a-zA-Z0-9'-]*_(VBD|VBG|VBN|VBP|VBZ|VB) [a-zA-Z0-9'-]*_(RBR|RBS|RB)|[a-zA-Z0-9'-]*_(VBD|VBG|VBN|VBP|VBZ|VB) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN)|[a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN) [a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN)|[a-zA-Z0-9'-]*_(NNPS|NNP|NNS|NN))"


def pos_tags(sentence):
    """Gets POS tags of given text"""
    sentence = re.sub("’", "'", sentence)
    tokens = nlp(sentence)
    tags = []
    for tok in tokens:
        tag = re.sub("\$","@",tok.tag_)
        tags.append(tok.text+"_"+tag)
    return " ".join(tags)


def extract_tagged_phrases(text):
    """Extracts tagged phrases using patterns"""
    found_iter = re.finditer(patterns, text)
    found = list(set([i.group() for i in found_iter]))
    return found


# import data
df = pd.read_csv("data/test.csv")

# tag the sentences of the msgs
df["Sentences"] = df["tweet"].apply(lambda x: PunktSentenceTokenizer().tokenize(str(x)))
s = df.apply(lambda x: pd.Series(x["Sentences"]),axis=1).stack().reset_index(level=1, drop=True)
s.name = "Sentence"
df = df.drop('Sentences', axis=1).join(s)
df = df.reset_index()
df["Tagged Sentence"] = df["Sentence"].apply(lambda x: pos_tags(str(x).lower()))

# get tagged phrases
df["Tagged Phrase List"] = df["Tagged Sentence"].apply(lambda x: extract_tagged_phrases(str(x)))
df = df[df["Tagged Phrase List"].str.len() != 0]
s = df.apply(lambda x: pd.Series(x["Tagged Phrase List"]),axis=1).stack().reset_index(level=1, drop=True)
s.name = "Tagged Phrase"
df = df.drop('Tagged Phrase List', axis=1).join(s)

# Get phrases
df["Phrase"] = df["Tagged Phrase"].apply(lambda x: re.sub("_[A-Z@]*","",x))

# Write phrases and their freq to file
phrases_dict = collections.Counter(df["Phrase"].tolist()).most_common()
f = open("out.csv", "w")
f.write("phrase\tfrequency\n")
for i in phrases_dict:
    f.write(i[0] + "\t" + str(i[1]) + "\n")
f.close()
