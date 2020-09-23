from sys import argv
from typing import Tuple

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np

MODEL_USED = ".models/entities_bert/"


# Returns 
def ner(text, model=MODEL_USED):
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_USED, do_lower_case=False, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_USED)

    # Generate label types
    labels = ["INSTRUMENT", "ORBIT", "DESIGN_ID", "INSTRUMENT_PARAMETER", "MEASUREMENT", "MISSION", "OBJECTIVE",
              "SPACE_AGENCY", "STAKEHOLDER", "SUBOBJECTIVE", "TECHNOLOGY", "NOT_PARTIAL_FULL", "NUMBER",
              "YEAR", "AGENT", "WAVEBAND"]

    label_types = ['O']
    for label in labels:
        label_types.append('B-' + label)
        label_types.append('I-' + label)

    # Create dicts for mapping from labels to IDs and back
    tag2idx = {t: i for i, t in enumerate(label_types)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    sentence_tokens_for_model = tokenizer(text,
                                          add_special_tokens=True,
                                          padding='max_length',
                                          truncation=True,
                                          max_length=128,
                                          return_token_type_ids=True,
                                          return_attention_mask=True,
                                          return_tensors='pt')

    sentence_tokens_for_postprocessing = tokenizer(text,
                                                   add_special_tokens=True,
                                                   padding='max_length',
                                                   truncation=True,
                                                   max_length=128,
                                                   return_token_type_ids=True,
                                                   return_attention_mask=True,
                                                   return_offsets_mapping=True)

    result: Tuple[Tensor] = model(**sentence_tokens_for_model)
    scores = result[0].detach().numpy()
    label_ids = np.argmax(scores, axis=2)
    predictions = [idx2tag[i] for i in label_ids[0]]

    # Converts tags to 1/word
    entities = []
    current_state = 'O'
    current_word = -1
    current_entity = None
    spans = sentence_tokens_for_postprocessing.data["offset_mapping"][1:]
    for idx, ent_tag in enumerate(predictions[1:]):
        word_idx = sentence_tokens_for_postprocessing.token_to_word(idx+1)
        if word_idx is None:
            break
        # Ignore all subwords when creating tags
        if current_word != word_idx:
            current_word = word_idx
            tag_info = ent_tag.split('-')
            if tag_info[0] in ['B', 'I']:
                # If beginning, start new entity and close last one
                if tag_info[0] == 'B':
                    if current_entity is not None:
                        entities.append(current_entity)
                    current_entity = [spans[idx][0], spans[idx][1], tag_info[1]]
                    current_state = tag_info[1]
                elif tag_info[0] == 'I' and tag_info[1] == current_state:
                    current_entity[1] = spans[idx][1]
            else:
                # Close current entity if exists
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
                current_state = 'O'
        else:
            if current_entity is not None:
                current_entity[1] = spans[idx][1]
    # Add last entity if it exists
    if current_entity is not None:
        entities.append(current_entity)

    print(text)
    for entity in entities:
        print(f"\"{text[entity[0]:entity[1]]}\" is a {entity[2]}")


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
