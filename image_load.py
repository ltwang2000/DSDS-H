import os
import torch
import numpy as np
from typing import Optional
import torch
import numpy as np
from typing import List


class GridImageDataset(torch.utils.data.Dataset):
    def __init__(self, feat_dir: str, split: str, image_ids: List[str]):
        # 修改后路径：直接使用feat_dir作为基础路径
        self.feat_dir = os.path.join(feat_dir, "features_grid", split)
        # 存储原始image_ids（带后缀）
        self.raw_image_ids = image_ids
        # 存储处理后的image_ids（去掉后缀）
        self.image_ids = [os.path.splitext(img_id)[0] for img_id in image_ids]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]  # 已经不带后缀
        file_path = os.path.join(self.feat_dir, f"{image_id}_grid.pth")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Grid特征不存在: {file_path}")
        data = torch.load(file_path, map_location="cpu")
        feat = data["feat"]

        return feat, None  # 确保返回(196,768)



class RegionImageDataset(torch.utils.data.Dataset):
    def __init__(self, feat_dir: str, split: str, image_ids: List[str]):
        self.feat_dir = os.path.join(feat_dir, "features_region", split)
        self.raw_image_ids = image_ids
        self.image_ids = [os.path.splitext(img_id)[0] for img_id in image_ids]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        # 修改为实际文件名格式（带_region.npz）
        file_path = os.path.join(self.feat_dir, f"{image_id}_region.npz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"[Region] Feature not found: {file_path}")

        npz_data = np.load(file_path)
        feat = torch.tensor(npz_data["feat"], dtype=torch.float32)

        return feat, None
