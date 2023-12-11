import nltk
from nltk.corpus import treebank
from spacy.training.example import Example
import spacy
import random
from tqdm import tqdm


# corpus = treebank.tagged_sents()
# tagged_sentences = list(corpus)
# filtered_data = [lst for lst in tagged_sentences if any(item[1] == "NNS" for item in lst)]
#TRAIN_DATA = [sublist for sublist in filtered_data if sublist]
# TRAIN_DATA = [[tup for tup in sublist if tup[0].is_punct()] for sublist in TRAIN_DATA]# suprrimer les vergulle ....
# TRAIN_DATA = [[tup for tup in sublist if tup[1].isalnum()] for sublist in TRAIN_DATA]# suprrimer les typpe que spacy ne connait pas comme -None
# TRAIN_DATA2=[]

# for sublist in TRAIN_DATA:
#     words = " ".join(word for word, tag in sublist)
#     tags = [tag for word, tag in sublist]
#     example = (words, tags)  
#     TRAIN_DATA2.append(example)

# print(TRAIN_DATA2[2])
# list = [item for sublist in TRAIN_DATA2 for item in sublist[1]]
# taggs = list(set(list))

# nlp = spacy.blank("en")
# pos_tagger = nlp.add_pipe("tagger")
# for tag in taggs:
#     pos_tagger.add_label(tag)
    
# other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'tagger']

# j=0
# epochs=50
# with nlp.disable_pipes(*other_pipes):
#     optimizer = nlp.begin_training()
#     for i in tqdm(range(epochs)):
#         random.shuffle(TRAIN_DATA2) 
#         for example in TRAIN_DATA2:
#             text, tags = example
#             try:
#                 example = Example.from_dict(nlp.make_doc(text.lower()), {"tags": tags})
#                 nlp.update([example],sgd=optimizer)
#             except ValueError as e:
#                 j+=1

# print("fin")
# #fin apprentisagge
# nlp.to_disk("NNS")

#descuter with monica

nlp = spacy.load("NNS")
text= ["i have many cows","animals are very beautifule ","we are humans we have many pieces of gold"]
for i,doc in enumerate(text):
    text[i]=doc.lower()

for txt in text:
    doc=nlp(txt)
    for x in doc:
        if not x.is_stop and not x.is_punct and not x.like_num:
            print(x.text,x.tag_)#spacy.explain(x)


# nlp.update(Examples)
# nlp.to_disk("NNS_model")






