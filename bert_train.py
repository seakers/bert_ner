import datetime
import json
import os

import spacy
from spacy.gold import biluo_tags_from_offsets
from transformers import TFBertForTokenClassification, BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

MODELS_DIR = "models"
DATASET_PATH = "data/EOSS_sentences"

nlp = spacy.load('en_core_web_sm')


def create_bert_data(json_data, tokenizer, tag2idx, idx2tag):
    sentences = []
    attention_masks = []
    labels = []
    # Pick only at most 2000 sentences from each data
    json_data = json_data[:500]
    for train_sentence in json_data:
        # 1. For each sentence, create BILUO tags
        spacy_doc = nlp(train_sentence[0])
        tags = biluo_tags_from_offsets(spacy_doc, [tuple(ner_tag) for ner_tag in train_sentence[1]['entities']])

        # 2. Use BERT tokenizer to get the tokenized words and tags
        tokenized_sentence = []
        tokenized_tags = []
        for idx, word in enumerate(spacy_doc):
            tokenized_word = tokenizer.tokenize(word.text)
            n_subwords = len(tokenized_word)

            tokenized_sentence.extend(tokenized_word)
            tokenized_tags.extend([tags[idx]] * n_subwords)

        # 3. BERT requirements
        inputs = tokenizer.encode_plus(tokenized_sentence, add_special_tokens=True, max_length=128)["input_ids"]
        tokenized_tags.insert(0, "O")
        attention_mask = [1] * len(inputs)

        sentences.append(inputs)
        attention_masks.append(attention_mask)
        labels.append([tag2idx.get(tag) for tag in tokenized_tags])

    # 4. Convert tokens and tags to IDs
    sentences = pad_sequences(sentences, maxlen=128, dtype='long', truncating='post', padding='post')
    attention_masks = pad_sequences(attention_masks, maxlen=128, dtype='long', truncating='post', padding='post')
    labels = pad_sequences(labels, maxlen=128, value=tag2idx["O"], padding='post', dtype='long', truncating='post')

    return sentences, attention_masks, labels


def get_train_set(tokenizer, tag2idx, idx2tag, path=DATASET_PATH):
    paths_list = [path + "/" + f_path for f_path in os.listdir(path)]
    train_sentences = []
    train_attention_masks = []
    train_labels = []
    for file_path in paths_list:
        with open(file_path, "r") as data_file:
            data = json.load(data_file)
            new_sentences, new_attention_masks, new_labels = create_bert_data(data, tokenizer, tag2idx, idx2tag)
            train_sentences.extend(new_sentences)
            train_attention_masks.extend(new_attention_masks)
            train_labels.extend(new_labels)
            print("File " + file_path + " processed")

    train_dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": train_sentences, "attention_mask": train_attention_masks}, train_labels))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(10000)
    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset


def main(model=None, new_model_name="entities_bert", models_dir=MODELS_DIR, n_iter=100):
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

    """Obtain Training Data"""
    TRAIN_DATA = get_train_set(tokenizer, tag2idx, idx2tag)

    """Set up the pipeline and entity recognizer, and train the new entity."""
    model = TFBertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_types))

    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_fn = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[acc_fn])

    os.makedirs(os.path.join(models_dir, new_model_name))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(models_dir, new_model_name, new_model_name+'_{epoch}.h5'),
            save_best_only=True,
            monitor='loss',
            verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    ]
    model.fit(TRAIN_DATA, epochs=3, callbacks=callbacks)


if __name__ == "__main__":
    main()
