import spacy
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora, models
from nltk.corpus import reuters

nlp = spacy.load("en_core_web_sm")

def preprocess(document):
    text = []
    doc = nlp(document.lower())
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.like_num:
            text.append(token.lemma_)
    return text

corpus_limit = 1000
reuters_subset = reuters.sents()[:corpus_limit]

# Process the subset of the corpus and extract bigrams and trigrams
processed_texts = [preprocess(' '.join(doc)) for doc in reuters_subset]

# Create bigram and trigram phrases
bigram = Phrases(processed_texts)
bigram_phraser = Phraser(bigram)
texts_with_bigrams = [bigram_phraser[text] for text in processed_texts]
print(texts_with_bigrams)


trigram = Phrases(texts_with_bigrams)
trigram_phraser = Phraser(trigram)
texts_with_trigrams = [trigram_phraser[text] for text in texts_with_bigrams]
print(texts_with_trigrams)

# Create a dictionary and bag-of-words corpus
dictionary = corpora.Dictionary(texts_with_trigrams)
corpus = [dictionary.doc2bow(text) for text in texts_with_trigrams]

# TF-IDF Model
tfidf = models.TfidfModel(corpus)
tfidf_corpus = [tfidf[doc] for doc in corpus]

word2vec = models.Word2Vec(sentences=texts_with_trigrams, vector_size=100, window=5, min_count=1, sg=0)

# Training the Word2Vec model
word2vec.train(texts_with_trigrams, total_examples=word2vec.corpus_count, epochs=10)
