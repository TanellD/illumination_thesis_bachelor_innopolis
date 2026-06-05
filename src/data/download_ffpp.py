#!/usr/bin/env python
""" Downloads FaceForensics++ and Deep Fake Detection public data release
Example usage:
    see -h or https://github.com/ondyari/FaceForensics

Thesis usage:
    # Download 100 videos at c23 compression from EU2 server
    python scripts/download_ffpp.py data_100 -d all -c c23 -n 100 --server EU2

    # Download full dataset
    python scripts/download_ffpp.py /data/faceforensics -d all -c c23 --server EU2

    # Download only original (real) videos
    python scripts/download_ffpp.py /data/faceforensics -d original -c c23 --server EU2

After downloading, update paths.yaml:
    data:
      ff_plus_plus: /path/to/your/download/dir
"""
# -*- coding: utf-8 -*-
import argparse
import os
import urllib
import urllib.request
import urllib.error
import tempfile
import time
import sys
import json
import random
from tqdm import tqdm
from os.path import join


# URLs and filenames
FILELIST_URL = 'misc/filelist.json'
DEEPFEAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'
DEEPFAKES_MODEL_NAMES = ['decoder_A.h5', 'decoder_B.h5', 'encoder.h5',]

# Parameters
DATASETS = {
    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',
    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
    }
ALL_DATASETS = ['original', 'DeepFakeDetection_original', 'Deepfakes',
                'DeepFakeDetection', 'Face2Face', 'FaceShifter', 'FaceSwap',
                'NeuralTextures']
COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks', 'models']
SERVERS = ['EU', 'EU2', 'CA']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics v2 public data release.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('output_path', type=str, help='Output directory.')
    parser.add_argument('-d', '--dataset', type=str, default='all',
                        help='Which dataset to download, either pristine or '
                             'manipulated data or the downloaded youtube '
                             'videos.',
                        choices=list(DATASETS.keys()) + ['all']
                        )
    parser.add_argument('-c', '--compression', type=str, default='raw',
                        help='Which compression degree. All videos '
                             'have been generated with h264 with a varying '
                             'codec. Raw (c0) videos are lossless compressed.',
                        choices=COMPRESSION
                        )
    parser.add_argument('-t', '--type', type=str, default='videos',
                        help='Which file type, i.e. videos, masks, for our '
                             'manipulation methods, models, for Deepfakes.',
                        choices=TYPE
                        )
    parser.add_argument('-n', '--num_videos', type=int, default=None,
                        help='Select a number of videos number to '
                             "download if you don't want to download the full"
                             ' dataset.')
    parser.add_argument('--server', type=str, default='EU',
                        help='Server to download the data from. If you '
                             'encounter a slow download speed, consider '
                             'changing the server.',
                        choices=SERVERS
                        )
    args = parser.parse_args()

    # URLs
    server = args.server
    if server == 'EU':
        server_url = 'http://canis.vc.in.tum.de:8100/'
    elif server == 'EU2':
        server_url = 'http://kaldir.vc.in.tum.de/faceforensics/'
    elif server == 'CA':
        server_url = 'http://falas.cmpt.sfu.ca:8100/'
    else:
        raise Exception('Wrong server name. Choices: {}'.format(str(SERVERS)))
    args.tos_url = server_url + 'webpage/FaceForensics_TOS.pdf'
    args.base_url = server_url + 'v3/'
    args.deepfakes_model_url = server_url + 'v3/manipulated_sequences/' + \
                               'Deepfakes/models/'

    return args


def download_file_with_retry(url, out_file_tmp, max_retries=10,
                              report_progress=False):
    """Download a file with exponential-backoff retry for network errors."""
    import socket
    last_exception = None

    for attempt in range(max_retries):
        try:
            if report_progress:
                urllib.request.urlretrieve(url, out_file_tmp,
                                           reporthook=reporthook)
            else:
                urllib.request.urlretrieve(url, out_file_tmp)
            return True
        except (urllib.error.URLError, ConnectionError, socket.gaierror) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt + random.uniform(0, 1), 60)
                print(f"\nDownload failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\nFailed after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            raise e

    raise last_exception


def download_files(filenames, base_url, output_path, report_progress=True):
    os.makedirs(output_path, exist_ok=True)
    if report_progress:
        filenames = tqdm(filenames)
    for filename in filenames:
        download_file(base_url + filename, join(output_path, filename))


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(
        "\rProgress: %d%%, %d MB, %d KB/s, %d seconds passed" %
        (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def download_file(url, out_file, report_progress=False):
    out_dir = os.path.dirname(out_file)
    if not os.path.isfile(out_file):
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()

        success = download_file_with_retry(url, out_file_tmp,
                                           max_retries=10,
                                           report_progress=report_progress)
        if success:
            os.rename(out_file_tmp, out_file)
        else:
            try:
                os.remove(out_file_tmp)
            except Exception:
                pass
            raise Exception(
                f"Failed to download {url} after multiple retries")
    else:
        tqdm.write('WARNING: skipping download of existing file ' + out_file)


def load_json_with_retry(url, max_retries=10):
    """Load JSON from URL with exponential-backoff retry."""
    import socket
    last_exception = None

    for attempt in range(max_retries):
        try:
            response = urllib.request.urlopen(url)
            return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionError, socket.gaierror) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = min(2 ** attempt + random.uniform(0, 1), 60)
                print(f"\nFailed to load JSON (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\nFailed to load JSON after {max_retries} attempts: {e}")
                raise last_exception
        except Exception as e:
            raise e

    raise last_exception


def main(args):
    # TOS
    print('By pressing any key to continue you confirm that you have agreed '
          'to the FaceForensics terms of use as described at:')
    print(args.tos_url)
    print('***')
    print('Press any key to continue, or CTRL-C to exit.')
    _ = input('')

    c_datasets = [args.dataset] if args.dataset != 'all' else ALL_DATASETS
    c_type = args.type
    c_compression = args.compression
    num_videos = args.num_videos
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    for dataset in c_datasets:
        dataset_path = DATASETS[dataset]

        if 'original_youtube_videos' in dataset:
            print('Downloading original youtube videos.')
            if 'info' not in dataset_path:
                print('Please be patient, this may take a while (~40gb)')
                suffix = ''
            else:
                suffix = 'info'
            download_file(
                args.base_url + '/' + dataset_path,
                out_file=join(output_path,
                              'downloaded_videos{}.zip'.format(suffix)),
                report_progress=True)
            return

        print('Downloading {} of dataset "{}"'.format(c_type, dataset_path))

        if 'DeepFakeDetection' in dataset_path or 'actors' in dataset_path:
            filepaths = load_json_with_retry(
                args.base_url + '/' + DEEPFEAKES_DETECTION_URL)
            if 'actors' in dataset_path:
                filelist = filepaths['actors']
            else:
                filelist = filepaths['DeepFakesDetection']
        elif 'original' in dataset_path:
            file_pairs = load_json_with_retry(
                args.base_url + '/' + FILELIST_URL)
            filelist = []
            for pair in file_pairs:
                filelist += pair
        else:
            file_pairs = load_json_with_retry(
                args.base_url + '/' + FILELIST_URL)
            filelist = []
            for pair in file_pairs:
                filelist.append('_'.join(pair))
                if c_type != 'models':
                    filelist.append('_'.join(pair[::-1]))

        if num_videos is not None and num_videos > 0:
            print('Downloading the first {} videos'.format(num_videos))
            filelist = filelist[:num_videos]

        dataset_videos_url = args.base_url + '{}/{}/{}/'.format(
            dataset_path, c_compression, c_type)
        dataset_mask_url = args.base_url + '{}/{}/videos/'.format(
            dataset_path, 'masks')

        if c_type == 'videos':
            dataset_output_path = join(output_path, dataset_path,
                                       c_compression, c_type)
            print('Output path: {}'.format(dataset_output_path))
            filelist = [filename + '.mp4' for filename in filelist]
            download_files(filelist, dataset_videos_url, dataset_output_path)

        elif c_type == 'masks':
            dataset_output_path = join(output_path, dataset_path,
                                       c_type, 'videos')
            print('Output path: {}'.format(dataset_output_path))
            if 'original' in dataset:
                if args.dataset != 'all':
                    print('Only videos available for original data. Aborting.')
                    return
                else:
                    print('Only videos available for original data. '
                          'Skipping original.\n')
                    continue
            if 'FaceShifter' in dataset:
                print('Masks not available for FaceShifter. Aborting.')
                return
            filelist = [filename + '.mp4' for filename in filelist]
            download_files(filelist, dataset_mask_url, dataset_output_path)

        else:
            if dataset != 'Deepfakes' and c_type == 'models':
                print('Models only available for Deepfakes. Aborting')
                return
            dataset_output_path = join(output_path, dataset_path, c_type)
            print('Output path: {}'.format(dataset_output_path))
            for folder in tqdm(filelist):
                folder_base_url = args.deepfakes_model_url + folder + '/'
                folder_dataset_output_path = join(dataset_output_path, folder)
                download_files(DEEPFAKES_MODEL_NAMES, folder_base_url,
                               folder_dataset_output_path,
                               report_progress=False)


if __name__ == "__main__":
    import socket
    args = parse_args()
    main(args)
