import spacy
from nltk.stem.porter import *
nlp = spacy.load("en_core_web_sm")

def get_classes(text):
    if not text.endswith("."): text += "."
    classes = set()
    business_env = ["database", "record", "system", "information", "organization", "detail", "website", "computer"]
    doc = nlp(text)
    skip_next = False
    for i, token in enumerate(doc):
        # Check if we need to skip the token
        if skip_next:
            skip_next = False
            continue
        # Check if the token is a noun and do not appear in business_env
        if token.pos_ == "NOUN" and token.text not in business_env:
            # Check if the next token is a noun (compound)
            if token.dep_ == "compound":
                classes.add(token.lemma_ + '_' + doc[i+1].lemma_)
                skip_next = True # Skip the next token

            # Check if the next token is a gerund
            elif doc[i+1].tag_ == "VBG":
                classes.add(token.lemma_ + '_' + doc[i+1].text)
                skip_next = True # Skip the next token

            # if the next token is nor a noun nor a gerund, add the token as a class
            else: classes.add(token.lemma_)
        
        # Check if the token is a gerund
        elif token.tag_ == "VBG":
            # Check if the next token is a noun
            if doc[i+1].pos_ == "NOUN":
                classes.add(token.text + '_' + doc[i+1].lemma_)
                skip_next = True # Skip the next token
    
    return classes

def get_attributes(text):
    doc = nlp(text)
    # attributes_noun_phrase_main = set()
    # relationship_attributes = set()
    concept_attributes = set()
    specific_indicators = {"type", "number", "date", "reference","no","code","volume","birth","id","address","name"}
    for i, token in enumerate(doc):
        # #A1
        # if token.tag_== "JJ":
        #     attributes_noun_phrase_main.add(token.lemma_)
        # #A2
        # if token.tag_ == "RB":
        #     if doc[i+1].pos_ == "VERB" and doc[i+1].pos_ != "ADJECTIVE":
        #         relationship_attributes.add(token.lemma_)
        #A3
        if token.tag_ == "POS":
            concept_attributes.add(doc[i+1].lemma_)
            if doc[i-2] == "and":
                concept_attributes.add(doc[i-3].lemma_)
        #A4
        if token.text == "of":
            concept_attributes.add(doc[i-1].lemma_)
        if token.tag_ == "NN" or token.tag_ == "NNS":
            if doc[i-1].tag_ == "IN":
                if doc[i-2].pos_ == "VERB":
                    concept_attributes.add(token.lemma_)
                    if i < len(doc):
                        if doc[i+1] == "and":
                            concept_attributes.add(doc[i+2].lemma_)
        #A5
        if token.text == "have" and doc[i-1].text == "to":
            concept_attributes.add(doc[i+1].lemma_)
        #A6
        if token in specific_indicators:
            concept_attributes.add(token.lemma_)
            if i < len(doc):
                if doc[i+1].text == "," and (doc[i+2].tag_ == "NN" or "NNS"):
                    concept_attributes.add(doc[i+2].lemma_)
                elif doc[i+1].tag_ == "NN" or "NNS":
                    concept_attributes.add(doc[i+1].lemma_)
    return concept_attributes #, relationship_attributes,attributes_noun_phrase_main

def get_subject_object(text):
    obj='none'
    sub='none'
    doc = nlp(text)
    for i,token in enumerate(doc):
        if(sub=='none'):
            if "subj" in token.dep_  and token.head.pos_ == "VERB": 
                if doc[i-1].dep_ == "compound":
                  sub= doc[i-1].lemma_ +'_'+ token.lemma_
                elif  "subj" in token.dep_ :
                  sub= token.lemma_
                    
        if(obj=='none'):
            if "dobj" in token.dep_ and token.head.pos_ == "VERB":
                if doc[i-1].dep_ == "compound":
                    obj= doc[i-1].lemma_ + '_' + token.lemma_
                elif  "dobj" in token.dep_ :
                    obj= token.lemma_
    entities = get_classes(text)
    if obj in entities and sub in entities:
         return sub,obj
    return 'none','none'
    

def get_relations(text):
    if not text.endswith("."): text += "." # Add a dot at the end of the sentence
    relations = []
    subjects_objects =[]
    verb1=['include','involve','contain','comprise','embrace']
    verb2=['consist of','divided to']
    verb_not_rel=['described', 'identified','characterized']
    doc = nlp(text)
    stemmer = PorterStemmer()
    skip_next = False
    for i, token in enumerate(doc):
        # Check if we need to skip the token
        if skip_next:
            skip_next = False
            continue
        # Check if the token is a verb
        if token.pos_ == "VERB" and token.text not in verb_not_rel:
            # check if next word is a preposition conjunction 
            if doc[i+1].text in ['by','in','on','to'] :
                # A verb followed by a preposition  can indicate a relations type
                    sub,obj=get_subject_object(text)
                    if obj!='none' and sub!='none':
                        relations.append(token.text+' '+doc[i+1].text)
                        subjects_objects.append((sub,obj))
                        skip_next = True # Skip the next token

            # if a verb is in the following list {include, involve, consists of, contain, comprise, divided to, embrace},
            #  this indicate a relations
            elif stemmer.stem(token.text) in verb1 : 
                    sub,obj=get_subject_object(text)
                    if obj!='none' and sub!='none':
                        relations.append(token.text)
                        subjects_objects.append((sub,obj))
                        skip_next = True # Skip the next token   

            elif stemmer.stem(token.text)+' '+doc[i+1].text in verb2 :
                    sub,obj=get_subject_object(text)
                    if obj!='none' and sub!='none':
                        relations.append(token.text+' '+doc[i+1].text)
                        subjects_objects.append((sub,obj))
                        skip_next = True # Skip the next token

            # check if verb is transitive
            elif doc[i+1].tag_ in ['NN','NNS'] or doc[i+1].tag_ =='DT' and doc[i+2].tag_ in ['NN','NNS']:
                #A transitive verb can indicate relations type
                    sub,obj=get_subject_object(text)
                    if obj!='none' and sub!='none':
                        relations.append(token.text)
                        subjects_objects.append((sub,obj))
                        skip_next = True # Skip the next token 

            elif doc[i+1].pos_ == 'ADJ' and doc[i+2].pos_ in ['NOUN','PROPN'] :
                    sub,obj=get_subject_object(text)
                    if obj!='none' and sub!='none':
                        relations.append(token.text)
                        subjects_objects.append((sub,obj))
                        skip_next = True

            elif doc[i+1].pos_ == 'DET' and doc[i+2].pos_ == 'ADJ' and doc[i+3].pos_ in ['NOUN','PROPN'] :
                    sub,obj=get_subject_object(text)
                    if obj!='none' and sub!='none':
                        relations.append(token.text)
                        subjects_objects.append((sub,obj))
                        skip_next = True
    return relations , subjects_objects

def get_subject_object_inh(text):
    obj='none'
    sub='none'
    doc = nlp(text)
    for i,token in enumerate(doc):
        if(sub=='none'):
            if "subj" in token.dep_  and token.head.text == "is": 
                if doc[i-1].dep_ == "compound":
                  sub= doc[i-1].lemma_ +'_'+ token.lemma_
                elif  "subj" in token.dep_ :
                  sub= token.lemma_
                    
        if(obj=='none'):
            if token.dep_ == 'attr' and token.head.text == "is":
                if doc[i-1].dep_ == "compound":
                    obj= doc[i-1].lemma_ + '_' + token.lemma_
                elif  "attr" in token.dep_ :
                    obj= token.lemma_
    entities = get_classes(text)
    if obj in entities and sub in entities:
         return sub,obj
    return 'none','none'
    
    
def get_inheritances(text):
    if not text.endswith("."): text += "." # Add a dot at the end of the sentence
    inheritance = []
    subjects_objects_inh =[]
    doc = nlp(text)
    skip_next = False

    for i, token in enumerate(doc):
        # Check if we need to skip the token
        if skip_next:
            skip_next = False
            continue

        if token.text in ["is","are"] and doc[i+1].text in ["a","an"] :
            sub,obj=get_subject_object_inh(text)
            if obj!='none' and sub!='none':
                inheritance.append(token.text+' '+doc[i+1].text)
                subjects_objects_inh.append((sub,obj))
                skip_next = True # Skip the next token
    return inheritance , subjects_objects_inh

text = "bottle opener"
classes = get_classes(text)
print(classes)
text = "a bottle can be oppened using a bottle opener"
classes = get_classes(text)
print(classes)
print(get_relations(text))
text = "bottle opener is used to open bottles"
classes = get_classes(text)
print(classes)
print(get_relations(text))
text = "question answering"
classes = get_classes(text)
print(classes)
text = "covering letter"
classes = get_classes(text)
print(classes)
text = "students write covering letter to apply for a job"
classes = get_classes(text)
print(classes)
print(get_relations(text))
text = "Every day, the mailman delivers registered mail in a geographical area assigned to him. The inhabitants are also associated with a geographical area. There are two types of registered mail: letters and parcels. As several letter carriers can intervene in the same area, we want, for each registered letter, the letter carrier who delivered it, in addition to the addressee"
classes = get_classes(text)
print(classes)
print(get_relations(text))