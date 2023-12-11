import spacy
from collections import Counter
import nltk

def transforme(doc, common_words, rare_words):
    tokens = [token.text for token in doc if token.text not in common_words and token.text not in rare_words and not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def lemmatisation(doc):
    tokens = [token.lemma_ for token in doc]
    return " ".join(tokens)

def steaming(doc):
    Porter = nltk.PorterStemmer()
    steam = [Porter.stem(terme) for terme in doc]
    return " ".join(steam)
    

NLP_ENG = spacy.load("en_core_web_sm")
#NLP_FR = spacy.load("fr_core_news_sm")
#NLP_ENG_EXTEND = spacy.load("en_core_web_md")
with open ("./alger.txt","r") as f:
    text=f.read()

doc=NLP_ENG(text.lower()) 
# print(steaming(doc))
word_frequencies = Counter(token.text for token in doc)
seuil1=3
most_common_words = [word for word,freq in word_frequencies.most_common(seuil1)]
seuil2 = 1
rare_words = [word for word, freq in word_frequencies.items()if freq <= seuil2]
NEW_DOC = NLP_ENG(transforme(doc, most_common_words, rare_words))# document transphormer
NEW_DOC = NLP_ENG(lemmatisation(NEW_DOC))  # limatiser
print(NEW_DOC)


