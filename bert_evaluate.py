from sys import argv

import spacy
from spacy.gold import spans_from_biluo_tags
from transformers import BertTokenizer, TFBertForTokenClassification
import tensorflow as tf
import numpy as np

MODEL_USED = "models/entities_bert/entities_bert_1.h5"
nlp = spacy.load('en_core_web_sm')

# Returns 
def ner(text, model=MODEL_USED):
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    # Generate label types
    labels = ["INSTRUMENT", "ORBIT", "DESIGN_ID", "INSTRUMENT_PARAMETER", "MEASUREMENT", "MISSION", "OBJECTIVE",
              "ORBIT", "SPACE_AGENCY", "STAKEHOLDER", "SUBOBJECTIVE", "TECHNOLOGY", "NOT_PARTIAL_FULL", "NUMBER",
              "YEAR", "AGENT"]

    label_types = ['O']
    for label in labels:
        label_types.append('B-' + label)
        label_types.append('I-' + label)
        label_types.append('L-' + label)
        label_types.append('U-' + label)

    # Create dicts for mapping from labels to IDs and back
    tag2idx = {t: i for i, t in enumerate(label_types)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    bert_model = TFBertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_types))
    bert_model.load_weights(model)

    encoded_text = tokenizer.encode(text)
    wordpieces = [tokenizer.decode(tok).replace(" ", "") for tok in encoded_text]
    scores = bert_model(tf.constant(encoded_text)[None, :])[0]
    label_ids = np.argmax(scores, axis=2)
    predictions = [idx2tag[i] for i in label_ids[0]]
    wp_preds = list(zip(wordpieces, predictions))
    toplevel_preds = [pair[1] for pair in wp_preds if "##" not in pair[0]]
    str_rep = " ".join([t[0] for t in wp_preds]).replace(" ##", "").split()

    doc = nlp(" ".join(str_rep[1:-1]))
    doc.ents = spans_from_biluo_tags(doc, toplevel_preds[1:-1])
    print(doc)
    for ent in doc.ents:
        print(ent, ent.label_)


if __name__ == "__main__":
    # Run test.py "text_to_evaluate- in the terminal
    text = argv[1]
    ner(text)
    # print("\nModel: {}\nText: {}\n".format(model, text))
    # nlp = spacy.load(model)
    # doc = nlp(text)
    #
    # # Find named entities, phrases and concepts
    # for index, entity in enumerate(doc.ents):
    #     print("{}: '{}' corresponds to {}".format(index + 1, entity.text, entity.label_))
    # print(doc)
