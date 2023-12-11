import spacy
from spacy.training.example import Example
import random
from tqdm import tqdm

train_data = [
    ("USTHB est une entreprise basée à Cupertino.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB est une universiter algeriene.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB se trouve à Bab El Oued.", {"entities": [(0, 5, "ORG")]}),
    ("L'université USTHB, située à Bab El Oued, est un établissement prestigieux.", {"entities": [(13, 18, "ORG")]}),
    ("USTHB offre des programmes de recherche avancée.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB a été fondée en 1974.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB est un établissement d'enseignement supérieur.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB est connue pour son excellence académique.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB a une longue histoire académique.", {"entities": [(0, 5, "ORG")]}),
    ("Fondée à Bab El Oued,l'USTHB est une institution académique respectée.", {"entities": [(23,28, "ORG")]}),
    ("USTHB est une université prestigieuse.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB propose des formations en sciences.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB est située à El Harrach.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB accueille des étudiants du monde entier.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB est un établissement d'enseignement supérieur de renommée mondiale.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB se distingue par ses programmes académiques de haute qualité.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB a contribué de manière significative à la recherche scientifique.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB est connue pour sa tradition académique d'excellence.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB est un acteur clé dans le domaine de l'éducation supérieure.", {"entities": [(0, 5, "ORG")]}),
    ("USTHB joue un rôle majeur dans la formation des futurs leaders.", {"entities": [(0, 5, "ORG")]}),
    ("LRIA est un laboratoire de recherche de pointe en intelligence artificielle.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA collabore avec des universités du monde entier.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA mène des projets de recherche novateurs en IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA est à la pointe de la recherche en intelligence artificielle.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA a une équipe de chercheurs hautement qualifiés.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA publie régulièrement des articles de recherche en IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA participe à des conférences internationales en IA.", {"entities": [(0, 4, "ORG")]}),
    ("Notre collaboration avec LRIA nous permet d'explorer de nouvelles frontières en ia .", {"entities": [(25, 29, "ORG")]}),
    ("LRIA a une réputation solide en matière de recherche en IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA est un acteur clé dans le domaine de l'IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA est impliqué dans des projets de pointe en IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA a des collaborations avec des entreprises innovantes.", {"entities": [(0, 4, "ORG")]}),
    ("L'Institut de Recherche en ia LRIA contribue grandement à l'avancement de l'IA.", {"entities": [(30,34, "ORG")]}),
    ("LRIA travaille sur des applications pratiques de l'IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA forme la prochaine génération de chercheurs en IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA a une longue histoire dans la recherche en IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA est un leader reconnu dans le domaine de l'IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA développe des solutions innovantes en IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA collabore avec des experts de renom en IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA est basé dans une ville réputée pour l'IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA s'engage à promouvoir la recherche en IA.", {"entities": [(0, 4, "ORG")]}),
    ("LRIA contribue activement à l'avancement de l'IA.", {"entities": [(0, 4, "ORG")]})
]

# nlp=spacy.load("en",disable=["ner"]) #charger un model avec notre ner modifier
# ner=nlp.create_pipe("ner")
# ner.from_disk("nom")
# nlp.add_pipe(ner,"nom")

nlp = spacy.load("fr_core_news_sm")
# nlp_updated = spacy.load("modele_bbz")

# # # Entraînement du modèle NER

epochs = 10

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for i in tqdm(range(epochs)):
        random.shuffle(train_data)       
        for text, annotations in train_data:
            example = Example.from_dict(nlp.make_doc(text.lower()), annotations)
            nlp.update([example],sgd=optimizer)
    #cette ligne ajoute des donner pour les utiliser dans al classification elle fait la classification avec les reseaux de neuron

nlp.to_disk("modele_bbz")


# ner = nlp.get_pipe('ner')
# ner.to_disk('naviner')  #pour sauvgarder uniquement le ner
# nlp.to_disk("modele_bbz") #pour sauvgarder tout le model 

nlp = spacy.load("fr_core_news_sm")

nlp = spacy.load("modele_bbz")

text= ["le drapeau est rouge","usthb est une entreprise basée à Alger","J'ai etudie a USTHB" ,"lria est formidable" ,"j'ai travailler a LRIA depuis 5 ans"]
for i,doc in enumerate(text):
    text[i]=doc.lower()
for txt in text:
    doc=nlp(txt)
    for x in doc:
        if not x.is_stop and not x.is_punct and not x.like_num:
            print(x.text,x.pos_,x.ent_type_)