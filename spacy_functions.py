import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import random
from pyUML import Graph, UMLClass
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
classes = set()

def get_classes(text):
    global classes
    classes = set()
    if not text.endswith("."): text += "."
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
    global classes
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
            classes.discard(doc[i+1].lemma_)
            if doc[i-2].text == "and":
                concept_attributes.add(doc[i-3].lemma_)
                classes.discard(doc[i-3].lemma_)
        # #A4
        # if token.text == "of":
        #     concept_attributes.add(doc[i-1].lemma_)
        #     classes.discard(doc[i-1].lemma_)
        if token.tag_ == "NN" or token.tag_ == "NNS":
            if doc[i-1].tag_ == "IN":
                if doc[i-2].pos_ == "VERB":
                    concept_attributes.add(token.lemma_)
                    classes.discard(token.lemma_)
                    if i < len(doc):
                        if doc[i+1].text == "and":
                            concept_attributes.add(doc[i+2].lemma_)
                            classes.discard(doc[i+2].lemma_)
        #A+
        if token.text == "by":
            if doc[i+1].pos_ == "PRON":
                concept_attributes.add(doc[i+2].lemma_)
                classes.discard(doc[i+2].lemma_)
                list__ = test_next_attr(doc, i+2)
                for l in list__:
                    concept_attributes.add(l)
                    classes.discard(l)
            else:
                concept_attributes.add(doc[i+1].lemma_)
                classes.discard(doc[i+1].lemma_)
                # list__ = test_next_attr(doc, i+2)
                # for l in list__:
                #     concept_attributes.add(l)
                #     classes.discard(l)
        #A5
        if token.text == "have" and doc[i-1].text == "to":
            concept_attributes.add(doc[i+1].lemma_)
            classes.discard(doc[i+1].lemma_)
        #A6
        if token.text in specific_indicators:
            concept_attributes.add(token.lemma_)
            classes.discard(token.lemma_)
            if i < len(doc):
                if doc[i+1].text == "," and (doc[i+2].tag_ == "NN" or "NNS"):
                    concept_attributes.add(doc[i+2].lemma_)
                    classes.discard(doc[i+2].lemma_)
                elif doc[i+1].tag_ == "NN" or "NNS":
                    concept_attributes.add(doc[i+1].lemma_)
                    classes.discard(doc[i+1].lemma_)
        ############################################################################
        # if token.text in concept_attributes:          
        #     list__ = test_next_attr(doc, i+2)
        #     for l in list__:
        #         concept_attributes.add(l)
        #         classes.discard(l)
    return concept_attributes #, relationship_attributes,attributes_noun_phrase_main

def test_next_attr(doc,i):
    list__ = set()
    br = 0
    while(br !=1):
        if doc[i+1].text == "," or ";":
            if doc[i+2].text == "and" and doc[i+3].pos_ == "PRON":
                list__.add(doc[i+4].lemma_)
            if doc[i+2].pos_ == "PRON":
                list__.add(doc[i+3].lemma_)
            else:
                list__.add(doc[i+2].lemma_)
            # if doc[i+2].pos_ != "PRON" and (doc[i+2].tag_ != "NN" or "NNS"):
            #     br = 1
        elif doc[i+1].text == "and":
            if doc[i+2].pos_ == "PRON":
                list__.add(doc[i+3].lemma_)
            else:
                list__.add(doc[i+2].lemma_)
        elif doc[i+1].pos_ == "PRON":
            list__.add(doc[i+2].lemma_)
        if doc[i+1].text == ".":
            br = 1
        i = i+1
    return list__

def get_subject_object(text,verb,index):
    # dependency markers for subjects
    SUBJECTS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
    # dependency markers for objects
    OBJECTS = {"dobj","dative", "attr", "oprd"}
    obj='none'
    sub='none'
    doc = nlp(text)
    for i in range(index-1,-1,-1):
        if doc[i].dep_ in SUBJECTS and doc[i].head.text == verb: 
            if doc[i-1].dep_ == "compound" and doc[i-2].dep_ == "compound":
                sub= doc[i-2].lemma_ +''+ doc[i-1].lemma_ +''+ doc[i].lemma_
                break
            elif doc[i-1].dep_ == "compound":
                sub= doc[i-1].lemma_ +''+ doc[i].lemma_
                break
            elif  doc[i].dep_ in SUBJECTS :
                sub= doc[i].lemma_
                break

    for i in range(index+1,len(doc)):    
        if doc[i].dep_ in OBJECTS and doc[i].head.text == verb:
            if doc[i-1].dep_ == "compound" and doc[i-2].dep_ == "compound":
                obj= doc[i-2].lemma_ +''+ doc[i-1].lemma_ +''+ doc[i].lemma_
                break
            elif doc[i-1].dep_ == "compound":
                obj=doc[i-1].lemma_+'' + doc[i].lemma_
                break
            elif  doc[i].dep_ in OBJECTS :
                obj= doc[i].lemma_
                break        
    # if we dont find subject and object so we will take it with the help of noun chunks
    if(obj == 'none' and sub == 'none'):
        for i in range(1,index+1):
            if obj!='none' : break
            if doc[index-i].pos_ in ['NOUN','PROPN']:
                if doc[index-i-1].dep_ != "compound":
                    obj=doc[index-i].lemma_
                else :
                    obj=doc[index-i-1].lemma_ +''+doc[index-i].lemma_
            elif doc[index-i].tag_ == 'MD':
                if doc[index-i-2].dep == "compound":
                    obj=doc[index-i-2].lemma_+''+doc[index-i-1].lemma_
                elif doc[index-i-1].pos_ in ['NOUN','PROPN']:
                    obj=doc[index-i-1].lemma_
    
        for i in range(1,len(doc)-index):
            if sub!='none' : break
            if doc[index+i].pos_ in ['NOUN','PROPN']:
                if doc[index+i].dep_ != "compound":
                    sub=doc[index+i].lemma_
                else :
                    sub=doc[index+i].lemma_ +''+doc[index+i+1].lemma_

            elif doc[index+i].pos_ in ['DET','ADJ']:
                if doc[index+i+1].pos_ in ['NOUN','PROPN'] :
                    if  doc[index+i+1].dep == "compound" :
                        sub=doc[index+i+1].lemma_ +''+doc[index+i+2].lemma_
                    else :
                        sub=doc[index+i+1].lemma_
    # if we dont find object so we will take the pobj of the verb
    i=index
    while i+1<len(doc) and obj=='none':
        if doc[i+1].dep_ =="pobj" and doc[i].head.text == verb:
            obj=doc[i+1].lemma_
        i+=1
    entities = get_classes(text)
    if obj in entities and sub in entities:
        return sub,obj
    return 'none','none'
    

def get_relations(text):
    if not text.endswith("."): text += "." # Add a dot at the end of the sentence
    subjects_verbs_objects =[]
    # verb_not_rel=['described', 'identified','characterized']
    verb_not_rel=[]
    doc = nlp(text)
    inheritance = []
    relationship = []
    objects = []
    object_inh = []
    for i, token in enumerate(doc):
        # check for inheritance
        if token.text in ["is", "are"] and doc[i+1].text in ["an", "a"]:
            sub, obj = get_subject_object(text, token.text, i)
            if obj != 'none' and sub != 'none':
                inheritance.append(token.text+' '+doc[i+1].text)
                object_inh.append((sub, obj))

        # Check if the token is a verb
        if token.pos_ == "VERB" and token.text not in verb_not_rel:
            sub,obj=get_subject_object(text,token.text,i)
            subjects_verbs_objects.append((sub,token.text,obj))
            relationship.append(token.text)
            objects.append((sub, obj))
    return inheritance, relationship, objects, object_inh

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

def text_to_uml(text):
    uml = {}
    entities = get_classes(text)
    attributes = get_attributes(text)
    for entity in entities:
        uml[entity] = []
    for attribute in attributes:
        entity = get_entity(text, attribute, entities)
        if entity:
            uml[entity].append((attribute, get_attribute_type(attribute)))

    inheritance, relationship, object, object_inh = get_relations(text)
    return uml, inheritance, relationship, object, object_inh

def get_attribute_type(attribute):
    ints = ["no", "number", "num", "nb", "age"]
    for i in ints:
        if i in attribute:
            return "int"
    if "date" in attribute:
      return "date"
    else:
      return "string"

def get_entity(text, attribute, entities):
    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(text)
    words_lem = []
    for word in words:
        words_lem.append(lemmatizer.lemmatize(word))
    i = words_lem.index(attribute)
    while i >= 0:
        word_lem = lemmatizer.lemmatize(words[i])
        if word_lem in entities:
            return word_lem
        i -= 1

chars = "abcdefghijklmnopqrstuvwxyz0123456789"
def get_random_id(length):
    return ''.join((random.choice(chars) for _ in range(length)))

def graph_from_uml(uml, inheritance, relationship, object, object_inh):
    graph = Graph('pyUML')
    for rel, obj in zip(relationship, object):
        lemmatizer = WordNetLemmatizer()
        class1 = UMLClass(lemmatizer.lemmatize(obj[0]))
        graph.add_class(class1)
        class2 = UMLClass(lemmatizer.lemmatize(obj[1]))
        graph.add_class(class1)
        # c1 = "0..*"
        # c2 = "0..*"
        graph.add_association(class1, class2, label=rel)
        # graph.add_association(class1, class2, label=rel, multiplicity_parent=c1, multiplicity_child=c2)

    for rel, obj in zip(inheritance, object_inh):
        class1 = UMLClass(obj[0])
        graph.add_class(class1)
        class2 = UMLClass(obj[1])
        graph.add_class(class1)
        graph.add_implementation(class1, class2)

    for entity in uml.keys():
        graph.add_class(UMLClass(entity, attributes={att[0]: att[1] for att in uml[entity] if len(uml[entity]) > 0}))

    return graph
