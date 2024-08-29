from io import BytesIO
import lmdb
import tifffile
from torch.utils.data import Dataset
import numpy as np
import random
import data.util as Util
import logging

class LRHRDataset(Dataset):
    def __init__(
        self,
        dataroot,
        datatype,
        l_resolution=16,
        r_resolution=128,
        split="train",
        data_len=-1,
        need_LR=False,
    ):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == "lmdb":
            self.env = lmdb.open(
                dataroot, readonly=True, lock=False, readahead=False, meminit=False
            )
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == "img":
            self.sr_path = Util.get_paths_from_images(
                "{}/sr_{}_{}".format(dataroot, l_resolution, r_resolution)
            )
            self.hr_path = Util.get_paths_from_images(
                "{}/hr_{}".format(dataroot, r_resolution)
            )
            # New paths for pre-fire Sentinel-2 images, Daymet, and LULC
            self.pre_fire_path = Util.get_paths_from_images(
                "{}/pre_fire_{}".format(dataroot, r_resolution)
            )
            self.daymet_path = Util.get_paths_from_images(
                "{}/Daymet_{}".format(dataroot, r_resolution)
            )
            self.lulc_path = Util.get_paths_from_images(
                "{}/LULC_{}".format(dataroot, r_resolution)
            )
            self.poly_path = Util.get_paths_from_images(
                "{}/hr_mask_{}".format(dataroot, r_resolution)
            )
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    "{}/lr_{}".format(dataroot, l_resolution)
                )

            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                "data_type [{:s}] is not recognized.".format(datatype)
            )

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == "lmdb":
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    "hr_{}_{}".format(self.r_res, str(index).zfill(5)).encode("utf-8")
                )
                sr_img_bytes = txn.get(
                    "sr_{}_{}_{}".format(
                        self.l_res, self.r_res, str(index).zfill(5)
                    ).encode("utf-8")
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        "lr_{}_{}".format(self.l_res, str(index).zfill(5)).encode(
                            "utf-8"
                        )
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len - 1)
                    hr_img_bytes = txn.get(
                        "hr_{}_{}".format(self.r_res, str(new_index).zfill(5)).encode(
                            "utf-8"
                        )
                    )
                    sr_img_bytes = txn.get(
                        "sr_{}_{}_{}".format(
                            self.l_res, self.r_res, str(new_index).zfill(5)
                        ).encode("utf-8")
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            "lr_{}_{}".format(
                                self.l_res, str(new_index).zfill(5)
                            ).encode("utf-8")
                        )
                img_HR = tifffile.imread(BytesIO(hr_img_bytes))
                img_SR = tifffile.imread(BytesIO(sr_img_bytes))
                if self.need_LR:
                    img_LR = tifffile.imread(BytesIO(lr_img_bytes))
        else:
            img_HR = tifffile.imread(self.hr_path[index])
            img_SR = tifffile.imread(self.sr_path[index])
            if self.need_LR:
                img_LR = tifffile.imread(self.lr_path[index])

        pre_fire_img = tifffile.imread(self.pre_fire_path[index])
        daymet_img = tifffile.imread(self.daymet_path[index])
        lulc_img = tifffile.imread(self.lulc_path[index])

        if self.need_LR:
            transformed_data = Util.transform_augment(
                [img_LR, img_SR, img_HR, pre_fire_img, daymet_img, lulc_img],
                split=self.split,
                min_max=(-1, 1),
            )
            img_LR, img_SR, img_HR, pre_fire_img, daymet_img, lulc_img,  = transformed_data
            if self.split == "val":
                return {
                    "LR": img_LR,
                    "HR": img_HR,
                    "SR": img_SR,
                    "Pre_Fire": pre_fire_img,
                    "Daymet": daymet_img,
                    "LULC": lulc_img,
                    "Index": index
                }
            return {
                "LR": img_LR,
                "HR": img_HR,
                "SR": img_SR,
                "Pre_Fire": pre_fire_img,
                "Daymet": daymet_img,
                "LULC": lulc_img,
                "Index": index
            }
        else:
            transformed_data = Util.transform_augment(
                [img_SR, img_HR, pre_fire_img, daymet_img, lulc_img],
                split=self.split,
                min_max=(-1, 1)
            )
            img_SR, img_HR, pre_fire_img, daymet_img, lulc_img = transformed_data
            return {
                "HR": img_HR,
                "SR": img_SR,
                "Pre_Fire": pre_fire_img,
                "Daymet": daymet_img,
                "LULC": lulc_img,
                "Index": index
            }