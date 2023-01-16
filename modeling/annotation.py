import spacy
import pandas as pd
import jsonlines

with jsonlines.open('admin.jsonl') as reader:
    json_var = [obj for obj in reader]

nlp = spacy.load("en_core_web_sm")

def doccano_ner_to_bert(json_var):
    texts = []
    poss = []
    labelss = []
    for i in range(len(json_var)):
        text = json_var[i]['text']
        labels = json_var[i]['entities']
        word_label_dict = {}
        for label in labels:
            word_label_dict[text[label["start_offset"]:label["end_offset"]]] = label["label"]
        
        doc = nlp(text)
        p = []
        bert_labels = []
        for word in doc:
            p.append(word.pos_)
            if word.text in list(word_label_dict.keys()):
                bert_labels.append(word_label_dict[word.text])
            else:
                bert_labels.append("O")

        texts.append(text)
        bert_labels = " ".join(bert_labels)
        labelss.append(bert_labels)
        p = " ".join(p)
        poss.append(p)
    return texts, labelss, poss

texts, labelss, poss = doccano_ner_to_bert(json_var)

df = pd.DataFrame({'req': texts, 'pos': poss, 'labels': labelss})
print(df)
# df.to_csv('manualy_annotated_dataset.csv', index=False)