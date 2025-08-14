import re
from tqdm import tqdm
import requests
import zipfile
from dadmatools.pipeline.utils.mwt_lemma_utils.seq2seq_utils import *
from dadmatools.pipeline.utils.mwt_lemma_utils.seq2seq_vocabs import *
import math
from numbers import Number
from .chuliu_edmonds import *
from .conll import *
from .tbinfo import *
from torch.utils.data import DataLoader, Dataset
import time
from datetime import datetime
import shutil
from .scorers import conll18_ud_eval as ud_eval
from huggingface_hub import hf_hub_download
import spacy
from spacy.tokens import Doc, Token, Span
from spacy.language import Language

SPACE_RE = re.compile(r'\s')


def trankit2conllu(trankit_output):
    assert type(trankit_output) == dict, "`trankit_output` must be a Python dictionary!"
    if SENTENCES in trankit_output and len(trankit_output[SENTENCES]) > 0 and TOKENS in trankit_output[SENTENCES][0]:
        output_type = 'document'
    elif TOKENS in trankit_output:
        output_type = 'sentence'
    else:
        print("Unknown format of `trankit_output`!")
        return None
    try:
        if output_type == 'document':
            json_doc = trankit_output[SENTENCES]
        else:
            assert output_type == 'sentence'
            json_doc = [trankit_output]

        conllu_doc = []
        for sentence in json_doc:
            conllu_sentence = []
            for token in sentence[TOKENS]:
                if type(token[ID]) == int or len(token[ID]) == 1:
                    conllu_sentence.append(token)
                else:
                    conllu_sentence.append(token)
                    for word in token[EXPANDED]:
                        conllu_sentence.append(word)
            conllu_doc.append(conllu_sentence)

        return CoNLL.dict2conllstring(conllu_doc)
    except:
        print('Unsuccessful conversion! Please check the format of `trankit_output`')
        return None


def remove_with_path(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def get_ud_score(system_conllu_file, gold_conllu_file):
    gold_ud = ud_eval.load_conllu_file(gold_conllu_file)
    system_ud = ud_eval.load_conllu_file(system_conllu_file)
    score = ud_eval.evaluate(gold_ud, system_ud)
    score['average'] = np.mean([v.f1 for v in list(score.values())])
    return score


def get_ud_performance_table(score):
    out = ''
    out += "Metric     | Precision |    Recall |  F1 Score | AligndAcc" + '\n'
    out += "-----------+-----------+-----------+-----------+-----------" + '\n'
    for metric in ["Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS",
                   "CLAS", "MLAS", "BLEX"]:
        out += "{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
            metric,
            100 * score[metric].precision,
            100 * score[metric].recall,
            100 * score[metric].f1,
            "{:10.2f}".format(100 * score[metric].aligned_accuracy) if score[
                                                                           metric].aligned_accuracy is not None else ""
        ) + '\n'
    return out


def unzip(dir, filename):
    with zipfile.ZipFile(os.path.join(dir, filename)) as f:
        f.extractall(dir)
    os.remove(os.path.join(dir, filename))


def download(cache_dir, language, saved_model_version, embedding_name):  # put a try-catch here
    lang_dir = os.path.join(cache_dir, embedding_name, language)
    save_fpath = os.path.join(lang_dir, '{}.zip'.format(language))

    if not os.path.exists(os.path.join(lang_dir, '{}.downloaded'.format(language))):
        # Updated to use correct UONLP repository on Hugging Face Hub
        url = "https://huggingface.co/uonlp/trankit/resolve/main/models/{}/{}/{}.zip".format(saved_model_version, embedding_name, language)
        print(url)

        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc='Downloading: ')

        ensure_dir(lang_dir)
        with open(save_fpath, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        unzip(lang_dir, '{}.zip'.format(language))
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("Failed to download saved models for {}!".format(language))
        else:
            with open(os.path.join(lang_dir, '{}.downloaded'.format(language)), 'w') as f:
                f.write('')


def download_hf(save_dir: str, pipelines):
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)

    hf_hub_download(repo_id=f"Dadmatech/Vocab", filename='persian.vocabs.json', local_dir=save_dir)
    hf_hub_download(repo_id=f"Dadmatech/Lemmatizer", filename='persian_lemmatizer.pt', local_dir=save_dir)
    hf_hub_download(repo_id=f"Dadmatech/mwt_expander", filename='persian_mwt_expander.pt', local_dir=save_dir)
    hf_hub_download(repo_id=f"Dadmatech/POS", filename='persian.tagger.mdl', local_dir=save_dir)
    hf_hub_download(repo_id=f"Dadmatech/tokenizer", filename='persian.tokenizer.mdl', local_dir=save_dir)

    if KASREH in pipelines:
        hf_hub_download(repo_id=f"Dadmatech/Kasreh_ezafe", filename='persian.kasreh.mdl', local_dir=save_dir)
        hf_hub_download(repo_id=f"Dadmatech/Kasreh_ezafe", filename='persian.kasreh-vocab.json', local_dir=save_dir)

    if NER in pipelines:
        hf_hub_download(repo_id=f"Dadmatech/NER", filename='persian.ner.mdl', local_dir=save_dir)
        hf_hub_download(repo_id=f"Dadmatech/NER", filename='persian.ner-vocab.json', local_dir=save_dir)

    if SENT in pipelines:
        hf_hub_download(repo_id=f"Dadmatech/Sent", filename='persian.sent.mdl', local_dir=save_dir)


def tget_output_doc(conllu_doc):
    out_doc = []
    num_sents = len(conllu_doc)
    for sent_id in range(num_sents):
        sent = conllu_doc[sent_id]
        out_sent = []

        start2mwt = {}
        for mwt in sent['mwts']:
            start2mwt[mwt['start']] = mwt

        num_words = len(sent.keys()) - 1
        for word_id in range(1, num_words + 1):
            word = sent[word_id]
            assert word['id'] == word_id
            if word_id in start2mwt:
                mwt = start2mwt[word_id]
                out_sent.append({ID: '{}-{}'.format(mwt['start'], mwt['end']), TEXT: mwt['text']})

            out_sent.append({ID: f'{word_id}', TEXT: word[TEXT],
                             UPOS: word.get(UPOS, '_'), XPOS: word.get(XPOS, '_'), FEATS: word.get(FEATS, '_'),
                             HEAD: word.get(HEAD, f'{word_id - 1}'), DEPREL: word.get(DEPREL, '_')})
        out_doc.append(out_sent)
    return out_doc


def get_output_doc(tokenized_doc, conllu_doc):
    num_sents = len(conllu_doc)
    for sent_id in range(num_sents):
        sent = conllu_doc[sent_id]
        out_sent = []

        start2mwt = {}
        for mwt in sent['mwts']:
            start2mwt[mwt['start']] = mwt

        num_words = len(sent.keys()) - 1
        for word_id in range(1, num_words + 1):
            word = sent[word_id]
            assert word['id'] == word_id
            if word_id in start2mwt:
                mwt = start2mwt[word_id]
                ori_tok = tokenized_doc[sent_id][TOKENS][len(out_sent)]
                out_sent.append({ID: (mwt['start'], mwt['end']), TEXT: mwt['text']})
                for k, v in ori_tok.items():
                    if k not in out_sent[-1]:
                        out_sent[-1][k] = v
            if not (len(out_sent) > 0 and type(out_sent[-1][ID]) == tuple and len(out_sent[-1][ID]) == 2 and word_id <=
                    out_sent[-1][ID][-1]):
                tmp = {ID: word_id, TEXT: word[TEXT]}
                if UPOS in word and word[UPOS] != '_':
                    tmp[UPOS] = word[UPOS]
                if XPOS in word and word[XPOS] != '_':
                    tmp[XPOS] = word[XPOS]
                if FEATS in word and word[FEATS] != '_':
                    tmp[FEATS] = word[FEATS]
                if HEAD in word and word[HEAD] != '_':
                    tmp[HEAD] = word[HEAD]
                if DEPREL in word and word[DEPREL] != '_':
                    tmp[DEPREL] = word[DEPREL]
                for k, v in tokenized_doc[sent_id][TOKENS][len(out_sent)].items():
                    if k not in tmp:
                        tmp[k] = v
                out_sent.append(tmp)
            else:
                expand_id = word_id - out_sent[-1][ID][0]
                if UPOS in word and word[UPOS] != '_':
                    out_sent[-1][EXPANDED][expand_id][UPOS] = word[UPOS]
                if XPOS in word and word[XPOS] != '_':
                    out_sent[-1][EXPANDED][expand_id][XPOS] = word[XPOS]
                if FEATS in word and word[FEATS] != '_':
                    out_sent[-1][EXPANDED][expand_id][FEATS] = word[FEATS]
                if HEAD in word and word[HEAD] != '_':
                    out_sent[-1][EXPANDED][expand_id][HEAD] = word[HEAD]
                if DEPREL in word and word[DEPREL] != '_':
                    out_sent[-1][EXPANDED][expand_id][DEPREL] = word[DEPREL]

        tokenized_doc[sent_id][TOKENS] = out_sent
    return tokenized_doc


def tget_output_sentence(sentence):
    sent = []
    i = 0
    for tok, wp_p, additional_info in sentence:
        if len(tok) <= 0:
            continue
        if wp_p == 3 or wp_p == 4:
            additional_info['MWT'] = 'Yes'
        infostr = None if len(additional_info) == 0 else '|'.join(
            [f"{k}={additional_info[k]}" for k in additional_info])
        sent.append({ID: i + 1, TEXT: tok})
        if infostr is not None: sent[-1][MISC] = infostr
        i += 1
    return sent


def get_output_sentence(sentence):
    sent = []
    i = 0
    start_position = sentence[0][2][DSPAN][0] if DSPAN in sentence[0][2] else sentence[0][2][SSPAN][0]
    for tok, wp_p, additional_info in sentence:
        if len(tok) <= 0:
            continue
        infostr = None
        if wp_p == 3 or wp_p == 4:
            additional_info['MWT'] = 'Yes'
            infostr = 'MWT=Yes'

        sent.append({ID: i + 1, TEXT: tok})
        if 'current_len' in additional_info:
            sent[-1][ID] += additional_info['current_len']

        for key, value in additional_info.items():
            if key != 'current_len':
                sent[-1][key] = value
        if SSPAN not in sent[-1]:
            sent[-1][SSPAN] = (sent[-1][DSPAN][0] - start_position, sent[-1][DSPAN][1] - start_position)
        if infostr is not None: sent[-1][MISC] = infostr
        i += 1
    return sent


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def word_lens_to_idxs_fast(token_lens):
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len


def compute_word_reps_avg(piece_reprs, component_idxs):
    batch_word_reprs = []
    batch_size, _, _ = piece_reprs.shape
    _, num_words, _ = component_idxs.shape
    for bid in range(batch_size):
        word_reprs = []
        for wid in range(num_words):
            wrep = torch.mean(piece_reprs[bid][component_idxs[bid][wid][0]: component_idxs[bid][wid][1]], dim=0)
            word_reprs.append(wrep)
        word_reprs = torch.stack(word_reprs, dim=0)  # [num words, rep dim]
        batch_word_reprs.append(word_reprs)
    batch_word_reprs = torch.stack(batch_word_reprs, dim=0)  # [batch size, num words, rep dim]
    return batch_word_reprs


def normalize_token(treebank_name, token, ud_eval=True):
    token = SPACE_RE.sub(' ', token.lstrip())

    if ud_eval:
        if 'chinese' in treebank_name.lower() or 'korean' in treebank_name.lower() or 'japanese' in treebank_name.lower():
            token = token.replace(' ', '')

    return token


def word_lens_to_idxs(word_lens):
    max_token_num = max([len(x) for x in word_lens])
    max_token_len = max([max(x) for x in word_lens])
    idxs = []
    for seq_token_lens in word_lens:
        seq_idxs = []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.append([offset, offset + token_len])
            offset += token_len
        seq_idxs.extend([[-1, 0]] * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
    return idxs, max_token_num, max_token_len


class Linears(nn.Module):
    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
        return inputs


def create_spacy_doc(dadmatools_output, nlp=None):
    """
    Convert DadmaTools output to a spaCy Doc object.
    
    Args:
        dadmatools_output: The output dictionary from DadmaTools pipeline
        nlp: spaCy language model (optional, will create a blank one if not provided)
    
    Returns:
        spacy.tokens.Doc: A spaCy Doc object with all the linguistic annotations
    """
    if nlp is None:
        # Create a blank spaCy model for Persian
        nlp = spacy.blank("fa")
    
    # Extract text and sentences from DadmaTools output
    text = dadmatools_output.get('text', '')
    sentences = dadmatools_output.get('sentences', [])
    
    if not text and sentences:
        # If no text provided, reconstruct from sentences
        text = ' '.join([' '.join([token.get('text', '') for token in sent.get('tokens', [])]) 
                        for sent in sentences])
    
    # Create spaCy Doc object
    doc = Doc(nlp.vocab, words=[], spaces=[])
    
    # Process each sentence and token
    all_tokens = []
    all_spaces = []
    current_pos = 0
    
    for sent_idx, sentence in enumerate(sentences):
        sent_tokens = sentence.get('tokens', [])
        
        for token_idx, token_data in enumerate(sent_tokens):
            token_text = token_data.get('text', '')
            
            # Add token to the document
            all_tokens.append(token_text)
            
            # Determine if there's a space after this token
            # Check if this is the last token in the sentence
            is_last_in_sentence = token_idx == len(sent_tokens) - 1
            is_last_sentence = sent_idx == len(sentences) - 1
            
            # Add space if not the last token in the document
            if not (is_last_in_sentence and is_last_sentence):
                all_spaces.append(True)
            else:
                all_spaces.append(False)
    
    # Create the Doc object with tokens and spaces
    doc = Doc(nlp.vocab, words=all_tokens, spaces=all_spaces)
    
    # Add linguistic annotations
    token_idx = 0
    for sent_idx, sentence in enumerate(sentences):
        sent_tokens = sentence.get('tokens', [])
        
        for token_data in sent_tokens:
            if token_idx < len(doc):
                token = doc[token_idx]
                
                # Add POS tag
                if 'upos' in token_data:
                    token.pos_ = token_data['upos']
                
                # Add detailed POS tag
                if 'xpos' in token_data:
                    token.tag_ = token_data['xpos']
                
                # Add morphological features
                if 'feats' in token_data:
                    # Convert string features to spaCy's MorphAnalysis format
                    from spacy.tokens import MorphAnalysis
                    token.morph = MorphAnalysis(token.vocab, token_data['feats'])
                
                # Add dependency information
                if 'head' in token_data and 'deprel' in token_data:
                    head_idx = token_data['head']
                    if isinstance(head_idx, int) and head_idx > 0:
                        # Adjust head index to account for 0-based indexing
                        adjusted_head = head_idx - 1
                        if adjusted_head < len(doc):
                            token.head = doc[adjusted_head]
                    token.dep_ = token_data['deprel']
                
                # Add lemma
                if 'lemma' in token_data:
                    token.lemma_ = token_data['lemma']
                
                # Add NER tag as custom extension
                if 'ner' in token_data:
                    ner_tag = token_data['ner']
                    token._.ner = ner_tag
                
                # Add custom extensions for DadmaTools specific annotations
                if 'kasreh' in token_data:
                    token._.kasreh = token_data['kasreh']
                
                token_idx += 1
    
    # Add custom attributes to the Doc object
    if 'lang' in dadmatools_output:
        doc._.lang = dadmatools_output['lang']
    
    if 'sentiment' in dadmatools_output:
        doc._.sentiment = dadmatools_output['sentiment']
    
    if 'spellchecker' in dadmatools_output:
        doc._.spellchecker = dadmatools_output['spellchecker']
    
    # Note: Sentence boundaries are not set as spaCy handles this automatically
    # based on the tokenization and punctuation
    
    return doc

def setup_spacy_extensions():
    """
    Set up custom spaCy extensions for DadmaTools specific annotations.
    This should be called once when the module is imported.
    """
    # Register custom attributes
    if not Token.has_extension("kasreh"):
        Token.set_extension("kasreh", default=None)
    
    if not Token.has_extension("ner"):
        Token.set_extension("ner", default=None)
    
    if not Doc.has_extension("lang"):
        Doc.set_extension("lang", default=None)
    
    if not Doc.has_extension("sentiment"):
        Doc.set_extension("sentiment", default=None)
    
    if not Doc.has_extension("spellchecker"):
        Doc.set_extension("spellchecker", default=None)

# Set up extensions when module is imported
setup_spacy_extensions()
