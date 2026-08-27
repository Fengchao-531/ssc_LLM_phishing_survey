import os
import pickle
import re

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from utils import sentence_tokenize


URL_PATTERNS = (
    re.compile(r"http[a-zA-Z0-9.?/&=:#%_-]*", re.S),
    re.compile(r"www.[a-zA-Z0-9.?/&=:#%_-]*", re.S),
    re.compile(r"[a-zA-Z0-9.?/&=:#%_-]*.(com|net|org|io|gov|me|edu)", re.S),
)


def _replace_urls(text):
    normalized = str(text)
    for pattern in URL_PATTERNS:
        normalized = pattern.sub(" <website> ", normalized)
    return normalized


def _tokenize_sentence(tokenizer, sentence, max_seq_len):
    tokens = tokenizer.tokenize(_replace_urls(sentence))
    if not tokens:
        return []
    packed = ["[CLS]"]
    packed.extend(tokens[: max_seq_len - 2])
    packed.append("[SEP]")
    return packed


def _build_doc_len(num_sentences, max_seq_num):
    if num_sentences <= 0:
        return [0]
    return [min(num_sentences - 1, max_seq_num - 1)]


def read_data(
    data_path,
    n_labeled_data=300,
    n_unlabeled_data=-1,
    max_seq_num=8,
    max_seq_len=64,
    embedding_size=128,
    bert_encoder=False,
):
    with open(os.path.join(data_path, "labeled_data.pkl"), "rb") as handle:
        labeled_data = pickle.load(handle)
    with open(os.path.join(data_path, "unlabeled_data.pkl"), "rb") as handle:
        unlabeled_data = pickle.load(handle)
    with open(os.path.join(data_path, "mid2target.pkl"), "rb") as handle:
        mid2target = pickle.load(handle)
    with open(os.path.join(data_path, "label_mapping.pkl"), "rb") as handle:
        label_mapping = pickle.load(handle)

    print(label_mapping)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    print("vocab size: ", vocab_size)

    np.random.seed(1)
    labeled_data_ids = list(labeled_data.keys())
    np.random.shuffle(labeled_data_ids)
    unlabeled_data_ids = list(unlabeled_data.keys())
    np.random.shuffle(unlabeled_data_ids)

    if len(labeled_data_ids) > 1000:
        n_labeled_data = min(len(labeled_data_ids) - 800, n_labeled_data)
    else:
        n_labeled_data = min(len(labeled_data_ids) - 500, n_labeled_data)

    train_labeled_ids = labeled_data_ids[:n_labeled_data]
    if n_unlabeled_data == -1:
        n_unlabeled_data = len(unlabeled_data_ids)
    train_unlabeled_ids = unlabeled_data_ids[:n_unlabeled_data]

    if len(labeled_data_ids) > 1000:
        val_ids = labeled_data_ids[-800:-400]
        test_ids = labeled_data_ids[-400:]
    else:
        val_ids = labeled_data_ids[-500:-300]
        test_ids = labeled_data_ids[-300:]

    train_labeled_dataset = Loader_labeled(
        tokenizer,
        labeled_data,
        train_labeled_ids,
        mid2target,
        label_mapping,
        max_seq_num,
        max_seq_len,
    )
    train_unlabeled_dataset = Loader_unlabeled(
        tokenizer,
        unlabeled_data,
        train_unlabeled_ids,
        mid2target,
        max_seq_num,
        max_seq_len,
    )
    val_dataset = Loader_labeled(
        tokenizer,
        labeled_data,
        val_ids,
        mid2target,
        label_mapping,
        max_seq_num,
        max_seq_len,
    )
    test_dataset = Loader_labeled(
        tokenizer,
        labeled_data,
        test_ids,
        mid2target,
        label_mapping,
        max_seq_num,
        max_seq_len,
    )

    n_class_sentence = len(set(label_mapping.values()))
    doc_labels = [value for value in mid2target.values()]
    n_class_doc = max(doc_labels) + 1

    print(
        "#Labeled: {}, Unlabeled {}, Val {}, Test {}, N class {}, {}".format(
            len(train_labeled_ids),
            len(train_unlabeled_ids),
            len(val_ids),
            len(test_ids),
            n_class_sentence,
            n_class_doc,
        )
    )

    return (
        train_labeled_dataset,
        train_unlabeled_dataset,
        val_dataset,
        test_dataset,
        vocab_size,
        n_class_sentence,
        n_class_doc,
    )


class Loader_labeled(Dataset):
    def __init__(
        self,
        tokenizer,
        labeled_data,
        ids,
        target,
        label_set,
        max_seq_num=8,
        max_seq_len=64,
    ):
        self.tokenizer = tokenizer
        self.ids = ids
        self.target = target
        self.max_seq_len = max_seq_len
        self.max_seq_num = max_seq_num
        self.label_set = label_set
        self.kde = KernelDensity(bandwidth=0.5, kernel="gaussian")
        self.load_data(labeled_data, ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        mid = self.ids[idx]
        sentences, labels, sent_len, doc_len = self.message[mid]

        message_target = self.lookup_score(mid)
        out_labels = np.array([10] * self.max_seq_num)
        out_doc_len = np.array(doc_len)
        sent_length = np.array([0] * self.max_seq_num)

        mask1 = np.array([0] * self.max_seq_num)
        mask2 = np.array([0] * self.max_seq_num)
        mask3 = np.array([1] * self.max_seq_num)
        mask4 = np.array([0] * self.max_seq_num)

        for index in range(0, len(labels)):
            out_labels[index] = labels[index]
            sent_length[index] = sent_len[index]
            if labels[index] != 10:
                mask1[index] = 1
                mask3[index] = 0
                mask4[index] = 1
            else:
                mask2[index] = 1
                mask3[index] = 0
                mask4[index] = 1

        message_vec = torch.LongTensor(self.message2id(sentences))
        return (
            message_vec,
            out_labels,
            message_target,
            mask1,
            mask2,
            mask3,
            mask4,
            mid,
            sent_length,
            out_doc_len,
        )

    def lookup_score(self, mid):
        return self.target[mid]

    def lookup_label_id(self, raw_label):
        return self.label_set[raw_label]

    def message2id(self, message):
        encoded = np.zeros([self.max_seq_num, self.max_seq_len])
        for sentence_index in range(0, len(message)):
            for token_index, token in enumerate(message[sentence_index]):
                if sentence_index < self.max_seq_num and token_index < self.max_seq_len:
                    encoded[sentence_index][token_index] = self.tokenizer._convert_token_to_id(token)

        for sentence_index in range(len(message), self.max_seq_num):
            encoded[sentence_index][0] = self.tokenizer._convert_token_to_id("[CLS]")
            encoded[sentence_index][1] = self.tokenizer._convert_token_to_id("[SEP]")

        return encoded

    def load_data(self, labeled_data, ids):
        self.message = {}
        label_samples = []

        for mid in tqdm(ids):
            sentences = []
            labels = []
            sent_len = []

            raw_sentences, raw_labels = labeled_data[mid]
            for sentence, raw_label in zip(raw_sentences, raw_labels):
                packed_tokens = _tokenize_sentence(self.tokenizer, sentence, self.max_seq_len)
                if not packed_tokens:
                    continue
                sentences.append(packed_tokens)
                mapped_label = self.lookup_label_id(raw_label)
                labels.append(mapped_label)
                label_samples.append(mapped_label)
                sent_len.append(len(packed_tokens) - 1)
                if len(sentences) >= self.max_seq_num:
                    break

            if not sentences:
                continue

            doc_len = _build_doc_len(len(sentences), self.max_seq_num)
            self.message[mid] = (sentences, labels, sent_len, doc_len)

        self.ids = [mid for mid in ids if mid in self.message]
        unique_labels = np.array(sorted(set(self.label_set.values())))
        self.kde.fit(np.array(label_samples)[:, None])
        self.dist = self.kde.score_samples(unique_labels[:, None])
        self.esit_dist = F.softmax(torch.tensor(self.dist), dim=-1)


class Loader_unlabeled(Dataset):
    def __init__(self, tokenizer, unlabeled_data, ids, target, max_seq_num=8, max_seq_len=64):
        self.tokenizer = tokenizer
        self.target = target
        self.max_seq_num = max_seq_num
        self.max_seq_len = max_seq_len
        self.load_data(unlabeled_data, ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        mid = self.ids[idx]
        sentences, labels, sent_len, doc_len = self.message[mid]

        message_target = self.lookup_score(mid)
        out_doc_len = np.array(doc_len)
        sent_length = np.array([0] * self.max_seq_num)
        out_labels = np.array([10] * self.max_seq_num)
        mask1 = np.array([0] * self.max_seq_num)
        mask2 = np.array([0] * self.max_seq_num)
        mask3 = np.array([1] * self.max_seq_num)
        mask4 = np.array([0] * self.max_seq_num)

        for index in range(0, len(labels)):
            out_labels[index] = labels[index]
            sent_length[index] = sent_len[index]
            if labels[index] != 10:
                mask1[index] = 1
                mask3[index] = 0
                mask4[index] = 1
            else:
                mask2[index] = 1
                mask3[index] = 0
                mask4[index] = 1

        message_vec = torch.LongTensor(self.message2id(sentences))
        return (
            message_vec,
            out_labels,
            message_target,
            mask1,
            mask2,
            mask3,
            mask4,
            mid,
            sent_length,
            out_doc_len,
        )

    def message2id(self, message):
        encoded = np.zeros([self.max_seq_num, self.max_seq_len])
        for sentence_index in range(0, len(message)):
            for token_index, token in enumerate(message[sentence_index]):
                if sentence_index < self.max_seq_num and token_index < self.max_seq_len:
                    encoded[sentence_index][token_index] = self.tokenizer._convert_token_to_id(token)

        for sentence_index in range(len(message), self.max_seq_num):
            encoded[sentence_index][0] = self.tokenizer._convert_token_to_id("[CLS]")
            encoded[sentence_index][1] = self.tokenizer._convert_token_to_id("[SEP]")

        return encoded

    def lookup_score(self, mid):
        return self.target[mid]

    def load_data(self, unlabeled_data, ids):
        self.message = {}
        self.ids = []

        for mid in tqdm(ids):
            try:
                doc = unlabeled_data[mid]
                if isinstance(doc, list):
                    raw_sentences = [str(value) for value in doc]
                else:
                    raw_sentences = sentence_tokenize(_replace_urls(str(doc) + "."))

                sentences = []
                labels = []
                sent_len = []
                for raw_sentence in raw_sentences:
                    packed_tokens = _tokenize_sentence(self.tokenizer, raw_sentence, self.max_seq_len)
                    if not packed_tokens:
                        continue
                    sentences.append(packed_tokens)
                    labels.append(10)
                    sent_len.append(len(packed_tokens) - 1)
                    if len(sentences) >= self.max_seq_num:
                        break

                if not sentences:
                    continue

                doc_len = _build_doc_len(len(sentences), self.max_seq_num)
                self.message[mid] = (sentences, labels, sent_len, doc_len)
                self.ids.append(mid)
            except Exception:
                pass
