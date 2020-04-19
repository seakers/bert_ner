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
    labels = []
    for train_sentence in json_data:
        # 1. For each sentence, create BILUO tags
        spacy_doc = nlp(train_sentence[0])
        tags = biluo_tags_from_offsets(spacy_doc, train_sentence[1]['entities'])

        # 2. Use BERT tokenizer to get the tokenized words and tags
        tokenized_sentence = []
        tokenized_tags = []
        for idx, word in enumerate(spacy_doc):
            tokenized_word = tokenizer.tokenize(word.text)
            n_subwords = len(tokenized_word)

            tokenized_sentence.extend(tokenized_word)
            tokenized_tags.extend([tags[idx]] * n_subwords)

        # 3. BERT requirements
        tokenized_sentence.insert(0, "[CLS]")
        tokenized_tags.insert(0, "O")

        # 4. Convert tokens and tags to IDs
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(tokenized_sentence)],
                                  maxlen=75, dtype='long', truncating='post', padding='post')
        tag_ids = pad_sequences([[tag2idx.get(tag) for tag in tokenized_tags]],
                                maxlen=75, value=tag2idx["O"], padding='post', dtype='long', truncating='post')
        sentences.append(input_ids)
        labels.append(tag_ids)

    return sentences, labels


def get_train_set(tokenizer, tag2idx, idx2tag, path=DATASET_PATH):
    paths_list = [path + "/" + f_path for f_path in os.listdir(path)]
    train_sentences = []
    train_labels = []
    for file_path in paths_list:
        with open(file_path, "r") as data_file:
            data = json.load(data_file)
            new_sentences, new_labels = create_bert_data(data, tokenizer, tag2idx, idx2tag)
            train_sentences.extend(new_sentences)
            train_labels.extend(new_labels)
            print("File " + file_path + " processed")

    train_dataset = tf.data.Dataset.from_tensor_slices((train_sentences, train_labels))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(10000)
    train_dataset = train_dataset.batch(64)
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
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    acc_fn = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[acc_fn])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(models_dir, new_model_name+'_{epoch}'),
            save_best_only=True,
            monitor='train_loss',
            verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), 'logs'))

    ]
    model.fit(TRAIN_DATA, epochs=5, callbacks=[callbacks])

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    # add new entity labels to entity recognizer
    ner.add_label("INSTRUMENT")
    ner.add_label("ORBIT")
    ner.add_label("DESIGN_ID")
    ner.add_label("INSTRUMENT_PARAMETER")
    ner.add_label("MEASUREMENT")
    ner.add_label("MISSION")
    ner.add_label("OBJECTIVE")
    ner.add_label("ORBIT")
    ner.add_label("SPACE_AGENCY")
    ner.add_label("STAKEHOLDER")
    ner.add_label("SUBOBJECTIVE")
    ner.add_label("TECHNOLOGY")
    ner.add_label("NOT_PARTIAL_FULL")
    ner.add_label("NUMBER")
    ner.add_label("YEAR")
    ner.add_label("AGENT")

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 20.0, 1.01)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            print("Losses", losses)

    # save model to output directory
    output_dir = models_dir + "/" + new_model_name
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


if __name__ == "__main__":
    main()
