import json
import os
from collections import namedtuple
from copy import deepcopy

import torch
from torch.utils.data import Dataset

# Define Instance and Batch structures for document processing
instance_fields = [
    'doc_index', 'text',
    'piece_idxs', 'attention_masks', 'word_lens',
    'label', 'label_id'
]

batch_fields = [
    'doc_index', 'texts',
    'piece_idxs', 'attention_masks', 'word_lens',
    'labels', 'label_ids', 'document_masks'
]

Instance = namedtuple('Instance', field_names=instance_fields)
Batch = namedtuple('Batch', field_names=batch_fields)

class DocumentDataset(Dataset):
    def __init__(self, config, documents):
        self.config = config
        self.wordpiece_splitter = config.wordpiece_splitter
        self.max_input_length = 512

        # Load documents
        self.data = []
        for doc in documents:
            # Directly store the entire text of the document
            text = doc['text']
            
            # Store the document information
            self.data.append({
                'doc_index': int(doc['index']),
                'text': text,
                'label': doc['label'],
                'label_id': int(doc['label_id'])
            })

        # # Adjust text handling if documents are too long
        # new_data = []
        # for doc in self.data:
        #     text = doc['text']
        #     tokens = text.split()  # Split text into tokens (or use another tokenizer if needed)
            
        #     # If text is too long, chunk it while keeping it as a single sequence
        #     if len(tokens) > self.max_input_length:
        #         sub_docs = []
        #         cur_text = ""
                
        #         for token in tokens:
        #             cur_text += " " + token
        #             if len(cur_text.split()) >= self.max_input_length:
        #                 sub_doc = deepcopy(doc)
        #                 sub_doc['text'] = cur_text.strip()
        #                 sub_docs.append(sub_doc)
        #                 cur_text = ""

        #         if cur_text:
        #             sub_doc = deepcopy(doc)
        #             sub_doc['text'] = cur_text.strip()
        #             sub_docs.append(sub_doc)

        #         new_data.extend(sub_docs)
        #     else:
        #         new_data.append(doc)

        # self.data = new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def numberize(self):
        # Prepare instances including piece indexes, attention masks, and word lengths
        data = []
        for doc in self.data:
            text = doc['text']
            # Tokenize the entire document as a single sequence of word pieces
            tokens = text.split()  # Basic tokenization, replace with a better tokenizer if needed
            pieces = [[p for p in self.wordpiece_splitter.tokenize(w) if p != '▁'] for w in tokens]
            for ps in pieces:
                if len(ps) == 0:
                    ps += ['-']
            word_lens = [len(x) for x in pieces]
            flat_pieces = [p for ps in pieces for p in ps]
            # Encode word pieces with special tokens
            piece_idxs = self.wordpiece_splitter.encode(
                flat_pieces,
                add_special_tokens=True,
                max_length=self.max_input_length,
                truncation=True
            )

            attn_masks = [1] * len(piece_idxs)
            
            instance = Instance(
                doc_index=doc['doc_index'],
                text=text,
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                word_lens=word_lens,
                label=doc['label'],
                label_id=doc['label_id']
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_doc_index = [inst.doc_index for inst in batch]
        batch_texts = [inst.text for inst in batch]
        batch_labels = [inst.label for inst in batch]
        batch_label_ids = [inst.label_id for inst in batch]

        max_piece_num = max(len(inst.piece_idxs) for inst in batch)

        batch_piece_idxs = []
        batch_attention_masks = []
        batch_word_lens = []
        batch_document_masks = []

        for inst in batch:
            # Padding documents
            batch_piece_idxs.append(inst.piece_idxs + [0] * (max_piece_num - len(inst.piece_idxs)))
            batch_attention_masks.append(inst.attention_masks + [0] * (max_piece_num - len(inst.attention_masks)))
            batch_word_lens.append(inst.word_lens)
            batch_document_masks.append([1] * len(inst.piece_idxs) + [0] * (max_piece_num - len(inst.piece_idxs)))

        batch_piece_idxs = torch.LongTensor(batch_piece_idxs).to(self.config.device)
        batch_attention_masks = torch.FloatTensor(batch_attention_masks).to(self.config.device)
        batch_document_masks = torch.LongTensor(batch_document_masks).eq(0).to(self.config.device)
        batch_label_ids = torch.LongTensor(batch_label_ids).to(self.config.device)

        return Batch(
            doc_index=batch_doc_index,
            texts=batch_texts,
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks,
            word_lens=batch_word_lens,
            labels=batch_labels,
            label_ids=batch_label_ids,
            document_masks=batch_document_masks
        )



