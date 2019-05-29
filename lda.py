import re
import html
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
import spacy
from nltk.tokenize import word_tokenize

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
other_stops = [ "http", "https"]
stop_words.extend(other_stops)

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def clean(text):
    # lowercase
    text = text.lower()
    # remove unicode strings
    text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    # replace html elements to standard words
    text = html.unescape(text)
    # remove email
    text = re.sub('\S*@\S*\s?', '', text)
    # remove newline chars
    text = re.sub('\s+', ' ', text)
    # remove stopwords
    words = [word for word in word_tokenize(text) if word.isalnum() and word not in stop_words]
    return " ".join(words)


# import dataset
df = pd.read_csv("data/test.csv")
df = df[["tweet"]]

# clean data and create a list
df["tweet"] = df["tweet"].apply(lambda x: clean(str(x)))
data = df.tweet.values
print(data)
# create list of list of words
data_words = list(sent_to_words(data))

# build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# form bigrams and trigrams
data_words_bigrams = make_bigrams(data_words)

# lemmatize keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# create dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Filter the terms which have occured in less than 100 msgs and more than 90% of the msgs
id2word.filter_extremes(no_below=10, no_above=0.9, keep_n=10000)

# create corpus
texts = data_lemmatized

# term document frequency
corpus = [id2word.doc2bow(text) for text in texts]

# build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=150,
                                           alpha='auto',
                                           per_word_topics=True)


# print the keyword in the 10 topics
pprint(lda_model.print_topics(num_words=10))


def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    sent_topics_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: x[1], reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data)

# format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# show
print(df_dominant_topic.head(10))
df_dominant_topic.to_csv("out.csv", index=False)


# number of documents for each topic
# topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
# print(topic_counts)