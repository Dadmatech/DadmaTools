import bz2
import glob
import json
import zipfile as zp
import tarfile
from tqdm import tqdm
import os
import sys
from pathlib import Path
import requests
import py7zr

DATASETS_INFO_ADDR = os.path.join(os.path.dirname(__file__), 'datasets_info.py')
DATASETS_DIR = os.path.join(os.path.dirname(__file__), 'datasets')
DATASET_INFO = json.load(open(DATASETS_INFO_ADDR, 'r'))
# DEFAULT_DESTINATION = os.path.join(str(Path(__file__).parent.absolute()).replace('/pipeline', ''), 'saved_models')
DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), '.dadmatools', 'datasets')
def unzip_dataset(from_path: str, to_path: str, zip_format=None) -> Path:
    """Unzip archive.
    Args:
        from_path (str): path of the archive
        to_path (str): path to the directory of extracted files

    Returns:
        path to the directory of extracted files
    """
    # extenstion = ''.join(Path(from_path).suffixes)

    extenstion = from_path
    if extenstion.endswith('zip') or zip_format == 'zip':
        with zp.ZipFile(from_path, 'r') as zfile:
            zfile.extractall(to_path)

    elif extenstion.endswith('.tar.gz') or extenstion.endswith('.tgz') or zip_format == 'gz':
        with tarfile.open(from_path, 'r:gz') as tgfile:
            for tarinfo in tgfile:
                tgfile.extract(tarinfo, to_path)
    elif extenstion.endswith('.tar.xz') or zip_format == 'xz':
        with tarfile.open(from_path) as f:
            for tarinfo in f:
                f.extract(tarinfo, to_path)
    elif extenstion.endswith('.bz2') or zip_format == 'bz2':
        zipfile = bz2.BZ2File(from_path)  # open the file
        data = zipfile.read()  # get the decompressed data
        newfilepath = from_path[:-4]  # assuming the filepath ends with .bz2
        open(newfilepath, 'wb').write(data)  # write a uncompressed file
    elif extenstion.endswith('7z') or zip_format == '7z':
        szfile = py7zr.SevenZipFile(from_path, mode='r')
        szfile.extractall(path=to_path)
        szfile.close()

    return Path(to_path)


def download_dataset(url, dest_dir, filename=None):
    # source_code: https://github.com/sirbowen78/lab/blob/master/file_handling/dl_file1.py
    # This example script downloads python program for mac.

    # Home directory of Mac, pathlib.Path module make this easy.
    # home_path = Path.home()
    # This is the sub directory under home directory.
    # sub_path = "tmp"
    # The header of the dl link has a Content-Length which is in bytes.
    # The bytes is in string hence has to convert to integer.

    os.makedirs(dest_dir, exist_ok=True)
    if 'drive.google' in url:
        import gdown
        # import os
        # print('gdown downloadddd output: ', dest_dir )
        # print(dest_dir, filename)
        # dest_dir = os.path.join(dest_dir,'peyma.zip')
        return gdown.download(url, quiet=False, output=filename)
    try:
        filesize = int(requests.head(url).headers["Content-Length"])
    except KeyError:
        print('unknown file length')
        filesize = -1
    # os.path.basename returns python-3.8.5-macosx10.9.pkg,
    # without this module I will have to manually split the url by "/"
    # then get the last index with -1.
    # Example:
    # url.split("/")[-1]
    filename = os.path.basename(url)

    # make the sub directory, exists_ok=True will not have exception if the sub dir does not exists.
    # the dir will be created if not exists.
    os.makedirs(dest_dir, exist_ok=True)

    # The absolute path to download the python program to.
    dl_path = os.path.join(dest_dir, filename)
    chunk_size = 1024
    if os.path.exists(dl_path):
        print(f'file {dl_path} already exist')
        return dl_path
    # Use the requests.get with stream enable, with iter_content by chunk size,
    # the contents will be written to the dl_path.
    # tqdm tracks the progress by progress.update(datasize)
    with requests.get(url, stream=True) as r, open(dl_path, "wb") as f, tqdm(
            unit="B",  # unit string to be displayed.
            unit_scale=True,  # let tqdm to determine the scale in kilo, mega..etc.
            unit_divisor=1024,  # is used when unit_scale is true
            total=filesize,  # the total iteration.
            file=sys.stdout,  # default goes to stderr, this is the display on console.
            desc=filename  # prefix to be displayed on progress bar.
    ) as progress:
        for chunk in r.iter_content(chunk_size=chunk_size):
            # download the file chunk by chunk
            datasize = f.write(chunk)
            # on each chunk update the progress bar.
            progress.update(datasize)

    return dl_path


def load_dataset_info(info_addr):
    with open(info_addr) as f:
        info = json.load(f)
    return info

def is_exist_dataset(dataset_info, dest_dir):
    return all(os.path.exists(os.path.join(dest_dir, fname)) for fname in dataset_info['filenames'])

def fill_datasets_info():
    datasets_info = {}
    for info_addr in glob.iglob(DATASETS_DIR + '/*/info.py'):
        ds_info = json.load(open(info_addr))
        datasets_info[ds_info['name']] = ds_info
    with open(DATASETS_INFO_ADDR, 'w+') as f:
        json.dump(datasets_info, f)


def get_all_datasets_info(tasks=None):
    datasets = DATASET_INFO
    if tasks:
        datasets = {ds:DATASET_INFO[ds] for ds in DATASET_INFO if DATASET_INFO[ds]['task'] in tasks}
    return datasets

def get_dataset_info(ds_name):
    if ds_name not in DATASET_INFO:
        raise  KeyError(f'{ds_name} not found in available datasets. call get_all_datasets_info() to see all available datasets')
    return DATASET_INFO[ds_name]


# if __name__ == '__main__':
#     fill_datasets_info()
