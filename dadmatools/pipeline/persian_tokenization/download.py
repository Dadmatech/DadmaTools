import urllib.request
import os
from tqdm import tqdm
from typing import Callable
import shutil
import gdown

MODELS = {
    'kasreh_ezafeh': {
        'url': 'https://drive.google.com/uc?id=12lp8MZvy840aPlyc9WggYm9HGOs46fjo&export=download&confirm=t',
        'file_extension': '.pt'
    },
    'spellchecker': {
        'url': 'https://drive.google.com/uc?id=1--rrJDfbxHwuausSnMlw81DI2t76uB2e&export=download&confirm=t',
        'file_extension': '.tar.gz'
    },
    'fa_lemmatizer': {
        'url': 'https://www.dropbox.com/s/2ne7bvkvsm97lzl/fa_ewt_lemmatizer.pt?dl=1',
        'file_extension': '.pt'
    },
    'fa_tokenizer': {
        'url': 'https://www.dropbox.com/s/bajpn68bp11o78s/fa_ewt_tokenizer.pt?dl=1',
        'file_extension': '.pt'
    },
    'fa_mwt': {
        'url': 'https://www.dropbox.com/s/9xqhfulttjlhv7u/fa_perdt_mwt_expander.pt?dl=1',
        'file_extension': '.pt'
    },
    'fa_constituency': {
        'url': 'https://www.dropbox.com/s/aro4jf544gf5pe3/const_model_tehran.pt?dl=1',
        'file_extension': '.pt'
    },
    'postagger': {
        'url': 'https://www.dropbox.com/s/vfrarh4wsvpfd7k/pos_model.pt?dl=1',
        'file_extension': '.pt'
    },
    'dependencyparser': {
        'url': 'https://www.dropbox.com/s/5t3lnxf0fz6020d/dependancy_model.pt?dl=1',
        'file_extension': '.pt'
    },
    'parsbert': {
        'url': 'https://www.dropbox.com/s/9p45owbt89zeyl9/parsbert.tar.gz?dl=1',
        'file_extension': '.tar.gz'
    },
    'ner': {
        'url': 'https://www.dropbox.com/s/xtb9x0fvm1tq3kk/NER.tar.gz?dl=1',
        'file_extension': '.tar.gz'
    }
}


class TqdmUpTo(tqdm):
    """
    This class provides callbacks/hooks in order to use tqdm with urllib.
    Read more here:
    https://github.com/tqdm/tqdm#hooks-and-callbacks
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def download_model(model_name: str, cache_dir: str, process_func: Callable = None,
                   clean_up_raw_data=True, force_download: bool = False, file_extension=None):
    if model_name not in MODELS:
        raise ValueError("The model {} do not exist".format(model_name))

    model_info = MODELS[model_name]
    model_info['name'] = model_name

    model_file = model_name + model_info['file_extension'] if not file_extension else model_name + file_extension
    model_file_path = os.path.join(cache_dir, model_file)

    if not os.path.exists(cache_dir): os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(model_file_path) or force_download:
        if 'drive.google' in model_info['url']:
            gdown.download(model_info['url'], model_file_path)
        else:
            _download_file(model_info, model_file_path)

    else:
        print("Model {} exists in {}".format(model_name, model_file_path))

    unzip_dir = os.path.join(cache_dir, model_name)
    if process_func is not None and not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir, exist_ok=True)
        process_func(model_file_path, model_info, cache_dir=cache_dir, unzip_dir=unzip_dir,
                     clean_up_raw_data=True)
    elif not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir, exist_ok=True)
        shutil.copy(model_file_path, unzip_dir)
    else:
        None

    return model_file_path


def _download_file(meta_info: dict, destination: str):
    url = meta_info['url']

    if not os.path.isfile(destination):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1) as t:
            t.set_description("Downloading file {}".format(destination))
            urllib.request.urlretrieve(url, destination, reporthook=t.update_to)
