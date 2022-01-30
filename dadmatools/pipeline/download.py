import urllib.request
import os
from pathlib import Path
from tqdm import tqdm
from typing import Callable
import shutil
import gdown

# DEFAULT_DESTINATION = os.path.join(str(os.getcwd()), 'saved_models')
DEFAULT_DESTINATION = os.path.join(str(Path(__file__).parent.absolute()).replace('/pipeline', ''), 'saved_models')
DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), '.pernlp')

MODELS = {
    'kasreh_ezafeh':{
        'url': 'https://drive.google.com/u/0/uc?id=1wyN7bHqSVnfZDBKHbACaU-7xmcs3Ofw_&export=download',
        'file_extension': '.pt'
    },
    'fa_lemmatizer':{
        'url': 'https://www.dropbox.com/s/2ne7bvkvsm97lzl/fa_ewt_lemmatizer.pt?dl=1',
        'file_extension': '.pt'
    },
    'fa_tokenizer':{
        'url': 'https://www.dropbox.com/s/bajpn68bp11o78s/fa_ewt_tokenizer.pt?dl=1',
        'file_extension': '.pt'
    },
    'fa_mwt':{
        'url': 'https://www.dropbox.com/s/9xqhfulttjlhv7u/fa_perdt_mwt_expander.pt?dl=1',
        'file_extension': '.pt'
    },
    'fa_constituency':{
        'url': 'https://www.dropbox.com/s/aro4jf544gf5pe3/const_model_tehran.pt?dl=1',
        'file_extension': '.pt'
    },
    'postagger':{
        'url': 'https://www.dropbox.com/s/vfrarh4wsvpfd7k/pos_model.pt?dl=1',
        'file_extension': '.pt'
    },
    'dependencyparser':{
        'url': 'https://www.dropbox.com/s/5t3lnxf0fz6020d/dependancy_model.pt?dl=1',
        'file_extension': '.pt'
    },
    'parsbert':{
        'url': 'https://www.dropbox.com/s/9p45owbt89zeyl9/parsbert.tar.gz?dl=1',
        'file_extension': '.tar.gz'
    },
    'ner':{
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
        
        
def download_model(model_name: str, cache_dir: str = DEFAULT_CACHE_DIR, process_func: Callable = None,
                   clean_up_raw_data=True, force_download: bool = False, file_extension=None):
    
    if model_name not in MODELS:
        raise ValueError("The model {} do not exist".format(model_name))

    model_info = MODELS[model_name]
    model_info['name'] = model_name

    model_file = model_name + model_info['file_extension'] if not file_extension else model_name + file_extension
    model_file_path = os.path.join(cache_dir, model_file)
    
    if not os.path.exists(cache_dir): os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(DEFAULT_DESTINATION): os.makedirs(DEFAULT_DESTINATION, exist_ok=True)
        
    if not os.path.exists(model_file_path) or force_download:
#         os.makedirs(cache_dir, exist_ok=True)

        single_file = model_info['name'] + model_info['file_extension']
        if 'drive.google' in model_info['url']:
            gdown.download(model_info['url'], model_file_path)
        else:
            _download_file(model_info, model_file_path)
    
    else:
        print("Model {} exists in {}".format(model_name, model_file_path))
    
    unzip_dir = os.path.join(DEFAULT_DESTINATION, model_name)
    if process_func is not None and not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir, exist_ok=True)
        process_func(model_file_path, model_info, cache_dir=DEFAULT_CACHE_DIR, unzip_dir = unzip_dir , clean_up_raw_data=True)
    elif not os.path.exists(unzip_dir):
        # moveing it to DEFAULT_DESTINATION
        os.makedirs(unzip_dir, exist_ok=True)
        shutil.copy(model_file_path, unzip_dir)
    else: None
    
    return model_file_path        



def _download_file(meta_info: dict, destination: str): 
    url = meta_info['url']
    model_name = meta_info['name']
    
    if not os.path.isfile(destination):
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1) as t:
            t.set_description("Downloading file {}".format(destination))
            urllib.request.urlretrieve(url, destination, reporthook=t.update_to)
            
    
    
def _unzip_process_func(tmp_file_path: str, meta_info: dict, unzip_dir: str, cache_dir: str = DEFAULT_CACHE_DIR, clean_up_raw_data: bool = True):
   
    
    import tarfile
    
    model_name = meta_info['name']
    full_path = os.path.join(cache_dir, model_name) + meta_info['file_extension']
    
    print("Unziping the file {}".format(model_name + meta_info['file_extension']))
    
    with tarfile.open(full_path, mode='r:gz', compresslevel=9) as tar:
        tar.extractall(path=unzip_dir)
            
            


# download_model('test', DEFAULT_CACHE_DIR, process_func=_unzip_process_func)
# download_model('fa_ewt_lemmatizer', DEFAULT_CACHE_DIR)

