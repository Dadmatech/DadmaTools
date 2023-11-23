from pathlib import Path
import time
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import os, sys
import pickle
import numpy as np
import transformers
import re
import shutil
from huggingface_hub import hf_hub_download
from transformers import BertConfig, AutoModel


def progressBar(value, endvalue, names, values, bar_length=30):
    assert (len(names) == len(values));
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    string = '';
    for name, val in zip(names, values):
        temp = '|| {0}: {1:.4f} '.format(name, val) if val != None else '|| {0}: {1} '.format(name, None)
        string += temp;
    sys.stdout.write("\rPercent: [{0}] {1}% {2}".format(arrow + spaces, int(round(percent * 100)), string))
    sys.stdout.flush()
    return


def load_data(base_path, corr_file, incorr_file):
    # load files
    if base_path:
        assert os.path.exists(base_path) == True
    incorr_data = []
    opfile1 = open(os.path.join(base_path, incorr_file), "r")
    for line in opfile1:
        if line.strip() != "": incorr_data.append(line.strip())
    opfile1.close()
    corr_data = []
    opfile2 = open(os.path.join(base_path, corr_file), "r")
    for line in opfile2:
        if line.strip() != "": corr_data.append(line.strip())
    opfile2.close()
    assert len(incorr_data) == len(corr_data)

    # verify if token split is same
    for i, (x, y) in tqdm(enumerate(zip(corr_data, incorr_data))):
        x_split, y_split = x.split(), y.split()
        try:
            assert len(x_split) == len(y_split)
        except AssertionError:
            print("# tokens in corr and incorr mismatch. retaining and trimming to min len.")

    # return as pairs
    data = []
    for x, y in tqdm(zip(corr_data, incorr_data)):
        data.append((x, y))

    return data


def batch_iter(data, batch_size, shuffle):
    """
    each data item is a tuple of lables and text
    """
    n_batches = int(np.ceil(len(data) / batch_size))
    indices = list(range(len(data)))
    if shuffle:  np.random.shuffle(indices)

    for i in range(n_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        batch_labels = [data[idx][0] for idx in batch_indices]
        batch_sentences = [data[idx][1] for idx in batch_indices]

        yield (batch_labels, batch_sentences)


def labelize(batch_labels, vocab):
    token2idx, pad_token, unk_token = vocab["token2idx"], vocab["pad_token"], vocab["unk_token"]
    list_list = [[token2idx[token] if token in token2idx else token2idx[unk_token] for token in line.split()] for line
                 in batch_labels]
    list_tensors = [torch.tensor(x) for x in list_list]
    tensor_ = pad_sequence(list_tensors, batch_first=True, padding_value=token2idx[pad_token])
    return tensor_, torch.tensor([len(x) for x in list_list]).long()


def tokenize(batch_sentences, vocab):
    token2idx, pad_token, unk_token = vocab["token2idx"], vocab["pad_token"], vocab["unk_token"]
    list_list = [[token2idx[token] if token in token2idx else token2idx[unk_token] for token in line.split()] for line
                 in batch_sentences]
    list_tensors = [torch.tensor(x) for x in list_list]
    tensor_ = pad_sequence(list_tensors, batch_first=True, padding_value=token2idx[pad_token])
    return tensor_, torch.tensor([len(x) for x in list_list]).long()


def untokenize(batch_predictions, batch_lengths, vocab):
    idx2token = vocab["idx2token"]
    unktoken = vocab["unk_token"]
    assert len(batch_predictions) == len(batch_lengths)
    batch_predictions = \
        [" ".join([idx2token[idx] for idx in pred_[:len_]]) \
         for pred_, len_ in zip(batch_predictions, batch_lengths)]
    return batch_predictions


def untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_clean_sentences, backoff="pass-through"):
    assert backoff in ["neutral", "pass-through"], print(f"selected backoff strategy not implemented: {backoff}")
    idx2token = vocab["idx2token"]
    unktoken = vocab["token2idx"][vocab["unk_token"]]
    assert len(batch_predictions) == len(batch_lengths) == len(batch_clean_sentences)
    batch_clean_sentences = [sent.split() for sent in batch_clean_sentences]
    if backoff == "pass-through":
        batch_predictions = \
            [" ".join([idx2token[idx] if idx != unktoken else clean_[i] for i, idx in enumerate(pred_[:len_])]) \
             for pred_, len_, clean_ in zip(batch_predictions, batch_lengths, batch_clean_sentences)]
    elif backoff == "neutral":
        batch_predictions = \
            [" ".join([idx2token[idx] if idx != unktoken else "a" for i, idx in enumerate(pred_[:len_])]) \
             for pred_, len_, clean_ in zip(batch_predictions, batch_lengths, batch_clean_sentences)]
    return batch_predictions


def untokenize_without_unks2(batch_predictions, batch_lengths, vocab, batch_clean_sentences, topk=None):
    """
    batch_predictions are softmax probabilities and should have shape (batch_size,max_seq_len,vocab_size)
    batch_lengths should have shape (batch_size)
    batch_clean_sentences should be strings of shape (batch_size)
    """
    idx2token = vocab["idx2token"]
    unktoken = vocab["token2idx"][vocab["unk_token"]]
    assert len(batch_predictions) == len(batch_lengths) == len(batch_clean_sentences)
    batch_clean_sentences = [sent.split() for sent in batch_clean_sentences]

    if topk is not None:
        # get topk items from dim=2 i.e top 5 prob inds
        batch_predictions = np.argpartition(-batch_predictions, topk, axis=-1)[:, :, :topk]  # (batch_size,max_seq_len,5)

    # get topk words
    idx_to_token = lambda idx, idx2token, corresponding_clean_token, unktoken: idx2token[
        idx] if idx != unktoken else corresponding_clean_token
    batch_predictions = \
        [[[idx_to_token(wordidx, idx2token, batch_clean_sentences[i][j], unktoken) \
           for wordidx in topk_wordidxs] \
          for j, topk_wordidxs in enumerate(predictions[:batch_lengths[i]])] \
         for i, predictions in enumerate(batch_predictions)]

    return batch_predictions


def get_model_nparams(model):
    ntotal = 0
    for param in list(model.parameters()):
        temp = 1
        for sz in list(param.size()): temp *= sz
        ntotal += temp
    return ntotal


def load_vocab_dict(path_: str):
    """
    path_: path where the vocab pickle file is saved
    """
    with open(path_, 'rb') as fp:
        vocab = pickle.load(fp)
    return vocab


def merge_subtokens(tokens: "list"):
    merged_tokens = []
    for token in tokens:
        if token.startswith("##"):
            merged_tokens[-1] = merged_tokens[-1] + token[2:]
        else:
            merged_tokens.append(token)
    text = " ".join(merged_tokens)
    return text


def _custom_bert_tokenize_sentence(BERT_TOKENIZER, BERT_MAX_SEQ_LEN, text):
    new_tokens = []
    tokens = BERT_TOKENIZER.tokenize(text)
    j = 0
    for i, t in enumerate(tokens):
        if t == '[UNK]':
            new_tokens.append(text.split()[j])
        else:
            new_tokens.append(t)
        if t[0] != '#':
            j += 1
    tokens = new_tokens
    tokens = tokens[:BERT_MAX_SEQ_LEN - 2]  # 2 allowed for [CLS] and [SEP]
    idxs = np.array([idx for idx, token in enumerate(tokens) if not token.startswith("##")] + [len(tokens)])
    split_sizes = (idxs[1:] - idxs[0:-1]).tolist()
    # NOTE: BERT tokenizer does more than just splitting at whitespace and tokenizing. So be careful.
    text = merge_subtokens(tokens)
    assert len(split_sizes) == len(text.split()), print(len(tokens), len(split_sizes), len(text.split()), split_sizes,
                                                        text)
    return text, tokens, split_sizes


def _custom_bert_tokenize_sentences(list_of_texts, BERT_TOKENIZER):
    out = [_custom_bert_tokenize_sentence(text=text, BERT_TOKENIZER=BERT_TOKENIZER, BERT_MAX_SEQ_LEN=512) for text in
           list_of_texts]
    texts, tokens, split_sizes = list(zip(*out))
    return [*texts], [*tokens], [*split_sizes]


def _simple_bert_tokenize_sentences(list_of_texts, BERT_TOKENIZER, BERT_MAX_SEQ_LEN):
    return [merge_subtokens(BERT_TOKENIZER.tokenize(text)[:BERT_MAX_SEQ_LEN - 2]) for text in
            list_of_texts]


def bert_tokenize(BERT_TOKENIZER, batch_sentences):
    """
    inputs:
        batch_sentences: List[str]
            a list of textual sentences to tokenized
    outputs:
        batch_attention_masks, batch_input_ids, batch_token_type_ids
            2d tensors of shape (bs,max_len)
        batch_splits: List[List[Int]]
            specifies #sub-tokens for each word in each textual string after sub-word tokenization
    """
    batch_sentences, batch_tokens, batch_splits = _custom_bert_tokenize_sentences(batch_sentences)

    batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens) for tokens in batch_tokens]

    batch_attention_masks = pad_sequence(
        [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
        padding_value=0)
    batch_input_ids = pad_sequence([torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts],
                                   batch_first=True, padding_value=0)
    batch_token_type_ids = pad_sequence(
        [torch.tensor(encoded_dict["token_type_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
        padding_value=0)

    batch_bert_dict = {"attention_mask": batch_attention_masks,
                       "input_ids": batch_input_ids,
                       "token_type_ids": batch_token_type_ids}

    return batch_sentences, batch_bert_dict, batch_splits


def bert_tokenize_for_valid_examples(BERT_TOKENIZER, batch_orginal_sentences, batch_noisy_sentences):
    """
    inputs:
        batch_noisy_sentences: List[str]
            a list of textual sentences to tokenized
        batch_orginal_sentences: List[str]
            a list of texts to make sure lengths of input and output are same in the seq-modeling task
    outputs (only of batch_noisy_sentences):
        batch_attention_masks, batch_input_ids, batch_token_type_ids
            2d tensors of shape (bs,max_len)
        batch_splits: List[List[Int]]
            specifies #sub-tokens for each word in each textual string after sub-word tokenization
    """
    _batch_orginal_sentences = _simple_bert_tokenize_sentences(batch_orginal_sentences, BERT_TOKENIZER,
                                                               BERT_MAX_SEQ_LEN=512)
    _batch_noisy_sentences, _batch_tokens, _batch_splits = _custom_bert_tokenize_sentences(batch_noisy_sentences,
                                                                                           BERT_TOKENIZER)
    valid_idxs = [idx for idx, (a, b) in enumerate(zip(_batch_orginal_sentences, _batch_noisy_sentences)) if
                  len(a.split()) == len(b.split())]
    batch_orginal_sentences = [line for idx, line in enumerate(_batch_orginal_sentences) if idx in valid_idxs]
    batch_noisy_sentences = [line for idx, line in enumerate(_batch_noisy_sentences) if idx in valid_idxs]
    batch_tokens = [line for idx, line in enumerate(_batch_tokens) if idx in valid_idxs]
    batch_splits = [line for idx, line in enumerate(_batch_splits) if idx in valid_idxs]

    batch_bert_dict = {"attention_mask": [], "input_ids": [], "token_type_ids": []}
    if len(valid_idxs) > 0:
        batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens) for tokens in batch_tokens]
        batch_attention_masks = pad_sequence(
            [torch.tensor(encoded_dict["attention_mask"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
            padding_value=0)
        batch_input_ids = pad_sequence(
            [torch.tensor(encoded_dict["input_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
            padding_value=0)
        batch_token_type_ids = pad_sequence(
            [torch.tensor(encoded_dict["token_type_ids"]) for encoded_dict in batch_encoded_dicts], batch_first=True,
            padding_value=0)
        batch_bert_dict = {"attention_mask": batch_attention_masks,
                           "input_ids": batch_input_ids,
                           "token_type_ids": batch_token_type_ids}

    return batch_orginal_sentences, batch_noisy_sentences, batch_bert_dict, batch_splits


def get_sentences_splitters(txt):
    splitters = ['? ', '! ', '. ', '.\n', '?\n', '!\n', ' ؟', '\n؟']
    all_sents = []
    last_sent_index = 0
    for i, (ch1, ch2) in enumerate(zip(txt, txt[1:])):
        if ch1 + ch2 in splitters:
            all_sents.append((txt[last_sent_index:i + len(ch1)], ch2))
            last_sent_index = i + len(ch1 + ch2)
    all_sents.append((txt[last_sent_index:], None))
    return [item[0] for item in all_sents], [item[1] for item in all_sents[:-1]]


def space_special_chars(txt):
    return re.sub('([.:،<>,!?()])', r' \1 ', txt)


def de_space_special_chars(txt):
    txt = re.sub('( ([.:،<>,!?()]) )', r'\2', txt)
    txt = re.sub('( ([.:،<>,!?()]))', r'\2', txt)
    return re.sub('(([.:،<>,!?()]) )', r'\2', txt)


class SubwordBert(nn.Module):
    def __init__(self, tokenizer_config_path, screp_dim, padding_idx, output_dim):
        super(SubwordBert, self).__init__()
        self.bert_dropout = torch.nn.Dropout(0.2)
        ###################
        ###################
        ###################
        ### here is the bug
        config = BertConfig.from_pretrained('Dadmatech/Nevise')
        ##################

        self.bert_model = AutoModel.from_config(config)
        self.bertmodule_outdim = self.bert_model.config.hidden_size

        # Uncomment to freeze BERT layers

        # output module
        assert output_dim > 0
        # self.dropout = nn.Dropout(p=0.4)
        self.dense = nn.Linear(self.bertmodule_outdim, output_dim)
        # loss
        # See https://pytorch.org/docs/stable/nn.html#crossentropyloss
        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_idx)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_merged_encodings(self, bert_seq_encodings, seq_splits, mode='avg'):
        bert_seq_encodings = bert_seq_encodings[:sum(seq_splits) + 2, :]  # 2 for [CLS] and [SEP]
        bert_seq_encodings = bert_seq_encodings[1:-1, :]
        # a tuple of tensors
        split_encoding = torch.split(bert_seq_encodings, seq_splits, dim=0)
        batched_encodings = pad_sequence(split_encoding, batch_first=True, padding_value=0)
        if mode == 'avg':
            seq_splits = torch.tensor(seq_splits).reshape(-1, 1).to(self.device)
            out = torch.div(torch.sum(batched_encodings, dim=1), seq_splits)
        elif mode == "add":
            out = torch.sum(batched_encodings, dim=1)
        else:
            raise Exception("Not Implemented")
        return out

    def forward(self,
                batch_bert_dict: "{'input_ids':tensor, 'attention_mask':tensor, 'token_type_ids':tensor}",
                batch_splits: "list[list[int]]",
                aux_word_embs: "tensor" = None,
                targets: "tensor" = None,
                topk=1):

        # cnn
        batch_size = len(batch_splits)

        # bert
        # BS X max_nsubwords x self.bertmodule_outdim
        bert_encodings, cls_encoding = self.bert_model(
            input_ids=batch_bert_dict["input_ids"],
            attention_mask=batch_bert_dict["attention_mask"],
            token_type_ids=batch_bert_dict["token_type_ids"],
            return_dict=False
        )
        bert_encodings = self.bert_dropout(bert_encodings)
        # BS X max_nwords x self.bertmodule_outdim
        bert_merged_encodings = pad_sequence(
            [self.get_merged_encodings(bert_seq_encodings, seq_splits, mode='avg') \
             for bert_seq_encodings, seq_splits in zip(bert_encodings, batch_splits)],
            batch_first=True,
            padding_value=0
        )

        # concat aux_embs
        # if not None, the expected dim for aux_word_embs: [BS,max_nwords,*]
        intermediate_encodings = bert_merged_encodings
        if aux_word_embs is not None:
            intermediate_encodings = torch.cat((intermediate_encodings, aux_word_embs), dim=2)

        # dense
        logits = self.dense(intermediate_encodings)

        # loss
        if targets is not None:
            assert len(targets) == batch_size  # targets:[[BS,max_nwords]
            logits_permuted = logits.permute(0, 2, 1)  # logits: [BS,output_dim,max_nwords]
            loss = self.criterion(logits_permuted, targets)

        # eval preds
        if not self.training:
            probs = F.softmax(logits, dim=-1)  # [BS,max_nwords,output_dim]
            if topk > 1:
                topk_values, topk_inds = \
                    torch.topk(probs, topk, dim=-1, largest=True,
                               sorted=True)  # -> (Tensor, LongTensor) of [BS,max_nwords,topk]
            elif topk == 1:
                topk_inds = torch.argmax(probs, dim=-1)  # [BS,max_nwords]

            # Note that for those positions with padded_idx,
            #   the arg_max_prob above computes a index because
            #   the bias term leads to non-uniform values in those positions

            return loss.cpu().detach().numpy(), topk_inds.cpu().detach().numpy()
        return loss


def model_inference(model, BERT_TOKENIZER, data, topk, DEVICE, BATCH_SIZE=16, vocab_=None):
    """
        model: an instance of SubwordBert
        data: list of tuples, with each tuple consisting of correct and incorrect
                sentence string (would be split at whitespaces)
        topk: how many of the topk softmax predictions are considered for metrics calculations
        """
    if vocab_ is not None:
        vocab = vocab_
    inference_st_time = time.time()
    _corr2corr, _corr2incorr, _incorr2corr, _incorr2incorr = 0, 0, 0, 0
    _mistakes = []
    VALID_BATCH_SIZE = BATCH_SIZE
    valid_loss = 0.
    data_iter = batch_iter(data, batch_size=VALID_BATCH_SIZE, shuffle=False)
    model.eval()
    model.to(DEVICE)
    results = []
    line_index = 0
    for batch_id, (batch_labels, batch_sentences) in tqdm(enumerate(data_iter)):
        torch.cuda.empty_cache()
        st_time = time.time()
        # set batch data for bert
        batch_labels_, batch_sentences_, batch_bert_inp, batch_bert_splits = bert_tokenize_for_valid_examples(
            BERT_TOKENIZER,
            batch_labels, batch_sentences)
        if len(batch_labels_) == 0:
            print("################")
            print("Not predicting the following lines due to pre-processing mismatch: \n")
            print([(a, b) for a, b in zip(batch_labels, batch_sentences)])
            print("################")
            continue
        else:
            batch_labels, batch_sentences = batch_labels_, batch_sentences_
        batch_bert_inp = {k: v.to(DEVICE) for k, v in batch_bert_inp.items()}
        # set batch data for others
        batch_labels_ids, batch_lengths = labelize(batch_labels, vocab)
        batch_lengths = batch_lengths.to(DEVICE)
        batch_labels_ids = batch_labels_ids.to(DEVICE)

        try:
            with torch.no_grad():
                """
                NEW: batch_predictions can now be of shape (batch_size,batch_max_seq_len,topk) if topk>1, else (batch_size,batch_max_seq_len)
                """
                batch_loss, batch_predictions = model(batch_bert_inp, batch_bert_splits, targets=batch_labels_ids,
                                                      topk=topk)
        except RuntimeError:
            print(f"batch_bert_inp:{len(batch_bert_inp.keys())},batch_labels_ids:{batch_labels_ids.shape}")
            raise Exception("")
        valid_loss += batch_loss
        batch_lengths = batch_lengths.cpu().detach().numpy()
        if topk == 1:
            batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_sentences)
        else:
            batch_predictions = untokenize_without_unks2(batch_predictions, batch_lengths, vocab, batch_sentences,
                                                         topk=None)
        batch_clean_sentences = [line for line in batch_labels]
        batch_corrupt_sentences = [line for line in batch_sentences]
        batch_predictions = [line for line in batch_predictions]

        for i, (a, b, c) in enumerate(zip(batch_clean_sentences, batch_corrupt_sentences, batch_predictions)):
            results.append({"id": line_index + i, "original": a, "noised": b, "predicted": c, "topk": [],
                            "topk_prediction_probs": [], "topk_reranker_losses": []})
        line_index += len(batch_clean_sentences)

        '''
        # update progress
        progressBar(batch_id+1,
                    int(np.ceil(len(data) / VALID_BATCH_SIZE)), 
                    ["batch_time","batch_loss","avg_batch_loss","batch_acc","avg_batch_acc"], 
                    [time.time()-st_time,batch_loss,valid_loss/(batch_id+1),None,None])
        '''
    return results


def load_bert_model(tokenizer_config_path, vocab):
    ## here  seems  to have bug
    model = SubwordBert(tokenizer_config_path, 3 * len(vocab["chartoken2idx"]), vocab["token2idx"][vocab["pad_token"]],
                        len(vocab["token_freq"]))
    return model


def load_pretrained(checkpoint_path, tokenizer_config_path, vocab, optimizer=None, device='cuda'):
    if torch.cuda.is_available() and device != "cpu":
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'
    state_dicts = torch.load(checkpoint_path, map_location=map_location)
    model = load_bert_model(tokenizer_config_path, vocab)
    model.load_state_dict(state_dicts, strict=False)
    return model


def load_pre_model(vocab_path, model_checkpoint_path, tokenizer_config_path):
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    vocab = load_vocab_dict(vocab_path)
    model = load_pretrained(model_checkpoint_path, tokenizer_config_path, vocab)
    return model, vocab, DEVICE


def spell_checking_on_sents(model, BERT_TOKENIZER, vocab, device, txt):
    txt = space_special_chars(txt)
    test_data = [(txt, txt)]
    greedy_results = model_inference(model, BERT_TOKENIZER, test_data, topk=1, DEVICE=device, BATCH_SIZE=1,
                                     vocab_=vocab)
    out = []
    for i, line in enumerate(greedy_results):
        ls = [(n, p) if n == p else ("**" + n + "**", "**" + p + "**") for n, p in
              zip(line["noised"].split(), line["predicted"].split())]
        y, z = map(list, zip(*ls))
        try:
            z = ' '.join(z)
            z = re.sub(r'\*\*(\w+)\*\*', r'** \1 **', z)
            z = re.sub(r'\*\* (\w+) \*\*', r'**\1**', z)
        except:
            z = ' '.join(z)
        out.append((" ".join(y), z))
    new_out = []
    for i, sent in enumerate(out):
        new_out.append((de_space_special_chars(out[i][0]), de_space_special_chars(out[i][1])))
    return new_out[0]


def get_config():
    config = {
        'save_model': 'spellchecker/state_dict_nevise.pt',
        'save_vocab': 'spellchecker/vocab.pkl',
        'config_tokenizer': 'spellchecker'
    }
    return config


def create_saved_model_dir(cache_dir: str):
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)


def download_from_hf(cache_dir: str):
    path = os.path.join(cache_dir, 'spellchecker')
    for filename in ["config.json", "state_dict_nevise.pt", "vocab.pkl", "vocab.txt"]:
        hf_hub_download(repo_id="Dadmatech/Nevise", filename=filename, local_dir=path)


def load_model(cache_dir: str):
    create_saved_model_dir(cache_dir)
    download_from_hf(cache_dir)

    config = get_config()

    model_path = f"{cache_dir}/{config['save_model']}"
    vocab_path = f"{cache_dir}/{config['save_vocab']}"
    tokenizer_config_path = f"{cache_dir}/{config['config_tokenizer']}"

    model, vocab, device = load_pre_model(vocab_path=vocab_path, model_checkpoint_path=model_path,
                                          tokenizer_config_path=tokenizer_config_path)
    BERT_TOKENIZER = transformers.BertTokenizer.from_pretrained(tokenizer_config_path,
                                                                do_lower_case=False)
    BERT_TOKENIZER.do_basic_tokenize = False
    BERT_TOKENIZER.tokenize_chinese_chars = False

    nlp = (model, vocab, device, BERT_TOKENIZER)

    return nlp


def spellchecker(nlp, sentence):
    model, vocab, device, BERT_TOKENIZER = nlp
    output = spell_checking_on_sents(model, BERT_TOKENIZER, vocab, device, sentence)
    checked = output[1].replace('*', '')
    return {'orginal': sentence, 'corrected': checked, 'checked_words': create_output(output)}

def create_output(output):
    pattern = r'\*\*[\w\s]+\*\*'
    mistakes = re.findall(pattern, output[0])
    checked = re.findall(pattern, output[1])
    return list(map(lambda x, y: (x.strip('*'), y.strip('*')), mistakes, checked))

# print(spellchecker(load_model(), 'جمله تستی برای نویسه اسست'))
# print(spellchecker(load_model(), 'واقعا حیف وقت که بنویسم سرویس دهیتون شده افتظاح'))
