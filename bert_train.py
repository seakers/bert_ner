import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple
import numpy as np

import torch
from filelock import FileLock
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, AutoConfig, TrainingArguments, \
    PreTrainedTokenizer, EvalPrediction
import tensorflow as tf
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

MODELS_DIR = "models"
DATASET_PATH = "./data/EOSS_sentences"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

@dataclass
class InputExample:
    """
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    sentence: str
    labels: Optional[Dict]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


def read_examples_from_file(data_dir: Path, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    paths_list = list(data_dir.glob("*.json"))
    guid_index = 1
    examples = []
    for file_path in paths_list:
        with file_path.open("r", encoding="utf-8") as data_file:
            data = json.load(data_file)
            for json_example in data:
                examples.append(InputExample(guid=f"{mode}-{guid_index}",
                                             sentence=json_example[0],
                                             labels=json_example[1]))
                guid_index += 1
            print("File " + file_path.name + " processed")
    return examples


def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        pad_token_label_id=-100,
) -> List[InputFeatures]:
    """Loads a data file into a list of `InputFeatures`
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        # First tokenize sentence and get splits
        sentence_tokens = tokenizer(example.sentence,
                                    add_special_tokens=True,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=max_seq_length,
                                    return_token_type_ids=True,
                                    return_attention_mask=True,
                                    return_offsets_mapping=True)

        # Fill label_ids
        entity_idx = 0
        current_state = "O"
        for offsets in sentence_tokens["offset_mapping"]:
            if offsets[0] == 0 and offsets[1] == 0:
                label_ids.append(pad_token_label_id)
                current_state = "O"
            elif entity_idx < len(example.labels["entities"]) and offsets[0] >= example.labels["entities"][entity_idx][0] and offsets[1] <= example.labels["entities"][entity_idx][1]:
                entity_label = example.labels['entities'][entity_idx][2]
                if current_state == "O":
                    label_ids.append(label_map[f"B-{entity_label}"])
                    current_state = entity_label
                elif current_state == entity_label:
                    label_ids.append(label_map[f"I-{entity_label}"])

                if offsets[1] == example.labels["entities"][entity_idx][1]:
                    entity_idx += 1
            else:
                label_ids.append(label_map["O"])
                current_state = "O"

        assert len(sentence_tokens["input_ids"]) == max_seq_length
        assert len(sentence_tokens["attention_mask"]) == max_seq_length
        assert len(sentence_tokens["token_type_ids"]) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in sentence_tokens["input_ids"]]))
            logger.info("input_mask: %s", " ".join([str(x) for x in sentence_tokens["attention_mask"]]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in sentence_tokens["token_type_ids"]]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=sentence_tokens["input_ids"],
                attention_mask=sentence_tokens["attention_mask"],
                token_type_ids=sentence_tokens["token_type_ids"],
                label_ids=label_ids
            )
        )
    return features


class NERDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
            self,
            data_dir: Path,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                examples = read_examples_from_file(data_dir, mode)
                # TODO clean up all this to leverage built-in features of tokenizers
                self.features = convert_examples_to_features(
                    examples,
                    labels,
                    max_seq_length,
                    tokenizer,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def create_bert_data(json_data, tokenizer, tag2idx, idx2tag):
    sentences = []
    attention_masks = []
    labels = []
    # Pick only at most 2000 sentences from each data
    json_data = json_data[:500]
    for train_sentence in json_data:
        # 1. For each sentence, create IOB tags
        spacy_doc = nlp(train_sentence[0])
        tags = biluo_tags_from_offsets(spacy_doc, [tuple(ner_tag) for ner_tag in train_sentence[1]['entities']])
        for idx, tag in enumerate(tags):
            tag_info = tag.split("-")
            if tag_info[0] == "U":
                tags[idx] = f"B-{tag_info[1]}"
            if tag_info[0] == "L":
                tags[idx] = f"I-{tag_info[1]}"

        # 2. Use BERT tokenizer to get the tokenized words and tags
        tokenized_sentence = []
        tokenized_tags = []
        for idx, word in enumerate(spacy_doc):
            tokenized_word = tokenizer.tokenize(word.text)
            n_subwords = len(tokenized_word)

            tokenized_sentence.extend(tokenized_word)
            tokenized_tags.extend([tags[idx]] * n_subwords)

        # 3. BERT requirements
        inputs = tokenizer.encode_plus(tokenized_sentence, add_special_tokens=True, max_length=128, truncation=True)["input_ids"]
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


def get_dataset(tokenizer, tag2idx, idx2tag, path=DATASET_PATH):
    paths_list = [Path(f"{path}/{f_path}") for f_path in os.listdir(path)]
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
            print("File " + file_path.name + " processed")

    train_dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": train_sentences, "attention_mask": train_attention_masks}, train_labels))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(10000)
    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset


def main(model=None, new_model_name="entities_bert", models_dir=MODELS_DIR, n_iter=100):
    # Generate label types
    labels = ["INSTRUMENT", "ORBIT", "DESIGN_ID", "INSTRUMENT_PARAMETER", "MEASUREMENT", "MISSION", "OBJECTIVE",
              "SPACE_AGENCY", "STAKEHOLDER", "SUBOBJECTIVE", "TECHNOLOGY", "NOT_PARTIAL_FULL", "NUMBER",
              "YEAR", "AGENT", "WAVEBAND"]

    label_types = ['O']
    for label in labels:
        label_types.append('B-' + label)
        label_types.append('I-' + label)

    # Create dicts for mapping from labels to IDs and back
    tag2idx: Dict[str, int] = {t: i for i, t in enumerate(label_types)}
    idx2tag: Dict[int, str] = {i: t for t, i in tag2idx.items()}

    num_labels = len(label_types)
    cache_folder = Path('./cache')
    config = AutoConfig.from_pretrained(
        'allenai/scibert_scivocab_cased',
        num_labels=num_labels,
        id2label=idx2tag,
        label2id=tag2idx,
        cache_dir=cache_folder,
    )

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        'allenai/scibert_scivocab_cased',
        do_lower_case=False,
        use_fast=True,
        cache_dir=cache_folder,
    )

    """Obtain Training Data"""
    ner_dataset = NERDataset(data_dir=Path(DATASET_PATH),
                             tokenizer=tokenizer,
                             labels=label_types,
                             max_seq_length=128,
                             overwrite_cache=False)
    eval_split = int(0.1*len(ner_dataset))
    train_dataset, eval_dataset = torch.utils.data.random_split(
        ner_dataset,
        [len(ner_dataset)-eval_split, eval_split],
        generator=torch.Generator().manual_seed(42)
    )

    """Set up the pipeline and entity recognizer, and train the new entity."""
    model = AutoModelForTokenClassification.from_pretrained(
        'allenai/scibert_scivocab_cased',
        config=config,
        cache_dir=cache_folder,
    )

    bert_model_dir = Path(f'./{models_dir}/{new_model_name}')
    bert_model_dir.mkdir(parents=True, exist_ok=True)
    bert_model_dir_str = bert_model_dir.resolve()
    bert_model_dir_str = str(bert_model_dir_str)
    # Initialize our Trainer
    training_args = TrainingArguments(
        output_dir=bert_model_dir_str,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        save_total_limit=5,
        save_steps=5000,
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(idx2tag[label_ids[i, j]])
                    preds_list[i].append(idx2tag[preds[i, j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
        return {
            "accuracy_score": accuracy_score(out_label_list, preds_list),
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(model_path=bert_model_dir_str)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)


if __name__ == "__main__":
    main()
