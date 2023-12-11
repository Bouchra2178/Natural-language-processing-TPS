import spacy
from spacy.training.example import Example
import random
from tqdm import tqdm
# def trouver_indices_mot(mot, chaine, label):
#     indices = []
#     indice_debut = chaine.find(mot)
    
#     while indice_debut != -1:
#         indice_fin = indice_debut + len(mot)
#         indices.append((indice_debut, indice_fin, label))
#         indice_debut = chaine.find(mot, indice_fin + 1)
    
#     return indices

# def chercher_et_modifier_couleurs(chaine):
#     # Liste des couleurs à rechercher
#     couleurs = ["bleu", "rouge", "jaune", "vert","noire", "rose", "turquoise", "orange", "blanc", "violet", "doré", "mosaïque", "beige", "rougeâtre", "brun", "multicolores"]
#     entites_couleur = []
    
#     for couleur in couleurs:
#         indices = trouver_indices_mot(couleur, chaine, "COLOR")
#         entites_couleur.extend(indices)
    
#     return chaine, entites_couleur

# chaines = [
#     "Le ciel était d'un bleu profond, et les fleurs étaient d'un rouge éclatant.",
#     "La pomme est rouge et délicieuse.",
#     "La forêt était d'un vert profond et mystérieux, cachant ses secrets dans l'ombre des arbres.",
#     "Le coucher de soleil peignait le ciel de nuances d'orange et de rose.",
#     "Le champ de blé ondulait sous le vent doré, éclatant de jaune.",
#     "La balle de tennis est jaune.",
#     "Les tulipes dans le jardin étaient d'un rouge vif, apportant une touche de couleur au paysage.",
#     "Le ciel nocturne était parsemé d'étoiles scintillantes, brillant comme des diamants.",
#     "La cascade rugissait, créant un mélange apaisant de bleu et de blanc.",
#     "La fumée est vert.",
#     "Les plumes du paon étaient d'un bleu électrique, un spectacle de couleurs éclatantes.",
#     "Les montagnes enneigées brillaient d'un blanc pur, contraste avec le ciel bleu glacial.",
#     "Les feuilles d'automne étaient d'un mélange de jaune, d'orange et de rouge, créant un tapis coloré.",
#     "Le coucher de soleil plongeait la ville dans une lueur doré.",
#     "Les rose du jardin sont rose.",
#     "La pluie tombait doucement, faisant ressortir le vert vif des feuilles des arbres.",
#     "Le tapis moelleux était d'un bleu profond, apportant une ambiance chaleureuse à la pièce.",
#     "La voiture de sport était d'un rouge vif, attirant tous les regards.",
#     "Le parapluie était d'un jaune éclatant, un point lumineux sous la pluie.",
#     "Le canapé en velours était d'un violet royal, ajoutant une touche d'élégance à la pièce.",
#     "Le lac était d'un bleu serein, reflétant le ciel sans nuages.",
#     "Le tournesol était d'un jaune vibrant, tournant son visage vers le soleil.",
#     "La mosaïque colorée sur le sol de la basilique était une œuvre d'art en soi.",
#     "Le chaton blanc était d'une douceur immaculée, ses yeux bleu comme des saphirs.",
#     "Les oiseaux dans le parc étaient d'un vert chatoyant, un spectacle de couleurs dans les arbres.",
#     "Le chapeau de paille était d'un beige naturel, parfait pour une journée ensoleillée.",
#     "Le papillon était d'un orange flamboyant, dansant parmi les fleurs.",
#     "La guitare vintage était d'un rouge profond, racontant une histoire à travers sa musique.",
#     "Le voilier était d'un blanc éclatant, glissant sur l'eau bleu.",
#     "Le feu de camp créait une lueur rouge dans la nuit noire.",
#     "Le livre ancien était d'un brun cuir, chargé d'histoires du passé.",
#     "Le drapeau national était d'un bleu, blanc et rouge patriotique.",
#     "La robe de mariée était d'un blanc immaculé, symbole de pureté et d'amour.",
#     "La tasse de café était d'un brun foncé, réchauffant les mains par une journée froide.",
#     "Le pommier était plein de pommes rouge prêtes à être récoltées.",
#     "La nuit étoilée était d'un bleu profond, la Voie lactée tracée à travers le ciel.",
#     "Les ballons multicolores flottaient dans le ciel, égayant la fête.",
#     "La licorne légendaire était d'un blanc étincelant, une créature mythique de rêve."
# ]

# train_data = []

# for chaine in chaines:
#     chaine_modifiee, entites_couleur = chercher_et_modifier_couleurs(chaine)
    
#     if entites_couleur:
#         resultat = (chaine_modifiee, {'entities': entites_couleur})
#         train_data.append(resultat)



# nlp = spacy.load("modele_bbz")
# epochs = 30

# other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

# with nlp.disable_pipes(*other_pipes):
#     optimizer = nlp.begin_training()
#     for i in tqdm(range(epochs)):
#         # random.shuffle(train_data)       
#         for text, annotations in train_data:
#             example = Example.from_dict(nlp.make_doc(text.lower()), annotations)
#             nlp.update([example],sgd=optimizer)
# nlp.to_disk("modele_color_bbz")

nlp=spacy.load("modele_color_bbz")
text= ["le drapeau est rouge","le ciel est vert","mon pull est rouge","usthb est bleu","usthb est une entreprise basée à Alger"]
for i,doc in enumerate(text):
    text[i]=doc.lower()
for txt in text:
    doc=nlp(txt)
    for x in doc:
        if not x.is_stop and not x.is_punct and not x.like_num:
            print(x.text,x.pos_,x.ent_type_)


