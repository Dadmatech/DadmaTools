import zipfile
import tarfile
from tqdm import tqdm
import os
import sys
from pathlib import Path
import requests


def unzip_archive(from_path: str, to_path: str, dest_filename) -> Path:
    """Unzip archive.
    Args:
        from_path (str): path of the archive
        to_path (str): path to the directory of extracted files

    Returns:
        path to the directory of extracted files
    """
    extenstion = ''.join(Path(from_path).suffixes)
    if extenstion.endswith('.zip'):
        with zipfile.ZipFile(from_path, 'r') as zfile:
            zfile.extractall(to_path)

    elif extenstion.endswith('.tar.gz') or extenstion.endswith('.tgz'):
        with tarfile.open(from_path, 'r:gz') as tgfile:
            for tarinfo in tgfile:
                tgfile.extract(tarinfo, to_path)
    elif extenstion.endswith('.gz'):
      import gzip
      import shutil
      out_file = os.path.join(to_path, dest_filename)
      with gzip.open(from_path, 'rb') as f_in:
        with open(out_file, 'wb') as f_out:
          shutil.copyfileobj(f_in, f_out)
    elif extenstion.endswith('.tar.xz'):
        with tarfile.open(from_path) as f:
            for tarinfo in f:
                f.extract(tarinfo, to_path)

    # elif '.tgz' '.7z':
    #     szfile = py7zr.SevenZipFile(from_path, mode='r')
    #     szfile.extractall(path=to_path)
    #     szfile.close()

    return Path(to_path)


def download_with_progress(url, dest_dir):
    # source_code: https://github.com/sirbowen78/lab/blob/master/file_handling/dl_file1.py
    # This example script downloads python program for mac.

    # Home directory of Mac, pathlib.Path module make this easy.
    # home_path = Path.home()
    # This is the sub directory under home directory.
    # sub_path = "tmp"
    # The header of the dl link has a Content-Length which is in bytes.
    # The bytes is in string hence has to convert to integer.
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