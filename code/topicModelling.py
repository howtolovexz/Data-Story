from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from math import sqrt
import pandas as pd
import gensim
from gensim import corpora
import pickle

def excludeLink(df):
    df = df.replace(regex=r"http\S+", value='')
    return df

colnames = ['created_at', 'text', 'screen_name', 'followers', 'friends', 'rt', 'fav', 'retweeted_status']
df = pd.read_csv('../../data/SampledData/worldcup20000samples.csv', names=colnames, header=0, sep=',')
# df = df.fillna(value=0)  # filled blank with zero
df = excludeLink(df) # remove link from the text

tokenized_text = []
tokenizer = RegexpTokenizer("\w+|%|-")
eng_stopwords = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Start pre-processing
tokenized_text = []
for text in df.text:
    # Lowercase
    text = str(text)
    lower_case = text.lower()

    # Tokenize
    tokens = tokenizer.tokenize(lower_case)

    # Re-initial token list in each round
    filtered_tokens = []

    # Remove stop word but include the negative helping verb
    for word in tokens:
        if not word in eng_stopwords:
        # Lemmatize
            lemmatized = lemmatizer.lemmatize(word, pos="v")
            filtered_tokens.append(lemmatized)

    # Append each tokenized tweet in the list
    tokenized_text.append(filtered_tokens)


dictionary = corpora.Dictionary(tokenized_text)
corpus = [dictionary.doc2bow(text) for text in tokenized_text]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')

NUM_TOPICS = 15
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=7)
for topic in topics:
    print(topic)


dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')
import pyLDAvis.gensim
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
lda_save = pyLDAvis.save_html(lda_display, '../../data/SampledData/test.html')