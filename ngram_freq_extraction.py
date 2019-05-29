import re
import ast
import collections
import pandas as pd
from nltk import ngrams
from nltk.corpus import stopwords


# stop_words = stopwords.words('english')
# stops.extend(["day","new","one","nan","people","get"])
regex_stop_words = "('i'|'me'|'my'|'myself'|'we'|'our'|'ours'|'ourselves'|'you'|'your'|'yours'|'yourself'|'yourselves'|'he'|'him'|'his'|'himself'|'she'|'her'|'hers'|'herself'|'it'|'its'|'itself'|'they'|'them'|'their'|'theirs'|'themselves'|'what'|'which'|'who'|'whom'|'this'|'that'|'these'|'those'|'am'|'is'|'are'|'was'|'were'|'be'|'been'|'being'|'have'|'has'|'had'|'having'|'do'|'does'|'did'|'doing'|'a'|'an'|'the'|'and'|'but'|'if'|'or'|'because'|'as'|'until'|'while'|'of'|'at'|'by'|'for'|'with'|'about'|'against'|'between'|'into'|'through'|'during'|'before'|'after'|'above'|'below'|'to'|'from'|'up'|'down'|'in'|'out'|'on'|'off'|'over'|'under'|'again'|'further'|'then'|'once'|'here'|'there'|'when'|'where'|'why'|'how'|'all'|'any'|'both'|'each'|'few'|'more'|'most'|'other'|'some'|'such'|'no'|'nor'|'not'|'only'|'own'|'same'|'so'|'than'|'too'|'very'|'s'|'t'|'can'|'will'|'just'|'don'|'should'|'now'|'d'|'ll'|'m'|'o'|'re'|'ve'|'y'|'ain'|'aren'|'couldn'|'didn'|'doesn'|'hadn'|'hasn'|'haven'|'isn'|'ma'|'mightn'|'mustn'|'needn'|'shan'|'shouldn'|'wasn'|'weren'|'won'|'wouldn'|'day'|'new'|'one'|'nan'|'people'|'get')"
num = 3 # no of ngrams to be extracted
n_most_common = 100

def clean_text(text):
    text = re.sub(r'(http(s)?:\/\/\S*?( |$)|www\.\S*?( |$))', " ", text)
    text = re.sub(r'([^\s\w]|_)+', '', text)
    return text


df = pd.read_csv("data/test.csv")
df["tweet"] = df["tweet"].apply(lambda x: clean_text(str(x).lower()))
texts = df["tweet"].tolist()
texts_str = " ".join(texts)

res = []
for i in range(num):
    ngrams_ = ngrams(texts_str.split(), i+1)
    temp = []
    for j in ngrams_:
        if not re.search(regex_stop_words, str(j)):
            temp.append(str(j))
    res.append(temp)


res_freq = []
for i in res:
    temp = []
    for j in collections.Counter(i).most_common(n_most_common):
        temp.append((" ".join(list(ast.literal_eval(j[0]))).strip(), j[1]))
    res_freq.append(temp)


df_dict = {}
for i in range(num):
    column = str(i+1) + "_ngram"
    column_freq = column + "_freq"
    ngram_list, ngram_freq = zip(*res_freq[i])
    df_dict[column] = ngram_list
    df_dict[column_freq] = ngram_freq


df_out = pd.DataFrame(df_dict)
df_out.to_csv("ngram_keywords.csv", index=False)