import argparse
from io import BytesIO
import multiprocessing
from multiprocessing import Lock, Process, RawValue
from functools import partial
from multiprocessing.sharedctypes import RawValue
import tifffile
from tqdm import tqdm
import os
from pathlib import Path
import lmdb
import numpy as np
import time


def resize_and_convert(img, size, resample):
    if img.shape[1] != size or img.shape[2] != size:  # Adjusted for multi-channel
        img = tifffile.resize(img, (img.shape[0], size, size), resample=resample)
    return img


def image_convert_bytes(img):
    buffer = BytesIO()
    tifffile.imwrite(buffer, img, photometric='rgb')  # Using tifffile to save the image
    return buffer.getvalue()


def resize_multiple(img, sizes=(16, 128), resample='bicubic', lmdb_save=False):
    lr_img = resize_and_convert(img, sizes[0], resample)
    hr_img = resize_and_convert(img, sizes[1], resample)
    sr_img = resize_and_convert(lr_img, sizes[1], resample)

    if lmdb_save:
        lr_img = image_convert_bytes(lr_img)
        hr_img = image_convert_bytes(hr_img)
        sr_img = image_convert_bytes(sr_img)

    return [lr_img, hr_img, sr_img]


def resize_worker(img_file, sizes, resample, lmdb_save=False):
    img = tifffile.imread(img_file)  # Using tifffile to read the image
    out = resize_multiple(
        img, sizes=sizes, resample=resample, lmdb_save=lmdb_save)

    return img_file.name.split('.')[0], out


class WorkingContext():
    def __init__(self, resize_fn, lmdb_save, out_path, env, sizes):
        self.resize_fn = resize_fn
        self.lmdb_save = lmdb_save
        self.out_path = out_path
        self.env = env
        self.sizes = sizes

        self.counter = RawValue('i', 0)
        self.counter_lock = Lock()

    def inc_get(self):
        with self.counter_lock:
            self.counter.value += 1
            return self.counter.value

    def value(self):
        with self.counter_lock:
            return self.counter.value


def prepare_process_worker(wctx, file_subset):
    for file in file_subset:
        i, imgs = wctx.resize_fn(file)
        lr_img, hr_img, sr_img = imgs
        if not wctx.lmdb_save:
            tifffile.imwrite(
                '{}/lr_{}/{}.tif'.format(wctx.out_path, wctx.sizes[0], i.zfill(5)), lr_img)
            tifffile.imwrite(
                '{}/hr_{}/{}.tif'.format(wctx.out_path, wctx.sizes[1], i.zfill(5)), hr_img)
            tifffile.imwrite(
                '{}/sr_{}_{}/{}.tif'.format(wctx.out_path,
                                            wctx.sizes[0], wctx.sizes[1], i.zfill(5)), sr_img)
        else:
            with wctx.env.begin(write=True) as txn:
                txn.put('lr_{}_{}'.format(
                    wctx.sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                txn.put('hr_{}_{}'.format(
                    wctx.sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                txn.put('sr_{}_{}_{}'.format(
                    wctx.sizes[0], wctx.sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
        curr_total = wctx.inc_get()
        if wctx.lmdb_save:
            with wctx.env.begin(write=True) as txn:
                txn.put('length'.encode('utf-8'), str(curr_total).encode('utf-8'))


def all_threads_inactive(worker_threads):
    for thread in worker_threads:
        if thread.is_alive():
            return False
    return True


def prepare(img_path, out_path, n_worker, sizes=(16, 128), resample='bicubic', lmdb_save=False):
    resize_fn = partial(resize_worker, sizes=sizes,
                        resample=resample, lmdb_save=lmdb_save)
    files = [p for p in Path(
        '{}'.format(img_path)).glob(f'**/*')]

    if not lmdb_save:
        os.makedirs(out_path, exist_ok=True)
        os.makedirs('{}/lr_{}'.format(out_path, sizes[0]), exist_ok=True)
        os.makedirs('{}/hr_{}'.format(out_path, sizes[1]), exist_ok=True)
        os.makedirs('{}/sr_{}_{}'.format(out_path,
                    sizes[0], sizes[1]), exist_ok=True)
    else:
        env = lmdb.open(out_path, map_size=1024 ** 4, readahead=False)

    if n_worker > 1:
        # prepare data subsets
        multi_env = None
        if lmdb_save:
            multi_env = env

        file_subsets = np.array_split(files, n_worker)
        worker_threads = []
        wctx = WorkingContext(resize_fn, lmdb_save, out_path, multi_env, sizes)

        # start worker processes, monitor results
        for i in range(n_worker):
            proc = Process(target=prepare_process_worker, args=(wctx, file_subsets[i]))
            proc.start()
            worker_threads.append(proc)
        
        total_count = str(len(files))
        while not all_threads_inactive(worker_threads):
            print("\r{}/{} images processed".format(wctx.value(), total_count), end=" ")
            time.sleep(0.1)

    else:
        total = 0
        for file in tqdm(files):
            i, imgs = resize_fn(file)
            lr_img, hr_img, sr_img = imgs
            if not lmdb_save:
                tifffile.imwrite(
                    '{}/lr_{}/{}.tif'.format(out_path, sizes[0], i.zfill(5)), lr_img)
                tifffile.imwrite(
                    '{}/hr_{}/{}.tif'.format(out_path, sizes[1], i.zfill(5)), hr_img)
                tifffile.imwrite(
                    '{}/sr_{}_{}/{}.tif'.format(out_path, sizes[0], sizes[1], i.zfill(5)), sr_img)
            else:
                with env.begin(write=True) as txn:
                    txn.put('lr_{}_{}'.format(
                        sizes[0], i.zfill(5)).encode('utf-8'), lr_img)
                    txn.put('hr_{}_{}'.format(
                        sizes[1], i.zfill(5)).encode('utf-8'), hr_img)
                    txn.put('sr_{}_{}_{}'.format(
                        sizes[0], sizes[1], i.zfill(5)).encode('utf-8'), sr_img)
            total += 1
            if lmdb_save:
                with env.begin(write=True) as txn:
                    txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        default='{}/Dataset/celebahq_256'.format(Path.home()))
    parser.add_argument('--out', '-o', type=str,
                        default='./dataset/celebahq')

    parser.add_argument('--size', type=str, default='64,512')
    parser.add_argument('--n_worker', type=int, default=3)
    parser.add_argument('--resample', type=str, default='bicubic')
    # default save in png format
    parser.add_argument('--lmdb', '-l', action='store_true')

    args = parser.parse_args()

    resample_map = {'bilinear': 'bilinear', 'bicubic': 'bicubic'}
    resample = resample_map[args.resample]
    sizes = [int(s.strip()) for s in args.size.split(',')]

    args.out = '{}_{}_{}'.format(args.out, sizes[0], sizes[1])
    prepare(args.path, args.out, args.n_worker,
            sizes=sizes, resample=resample, lmdb_save=args.lmdb)
