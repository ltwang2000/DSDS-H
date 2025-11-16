import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import roi_align

# 图像 split 映射
image_dirs = {
    'train': 'train',
    'valid': 'test_2016_val',
    'test2016': 'test_2016_flickr',
    'test2017': 'test2017-images',
    'testcoco': 'testcoco-images',
}

split_files = {
    'train': 'train.txt',
    'valid': 'valid.txt',
    'test2016': 'test_2016_flickr.txt',
    'test2017': 'test_2017_flickr.txt',
    'testcoco': 'test_2017_mscoco.txt',
}

def get_filenames(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip().split('#')[0] for line in f]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=image_dirs.keys(), required=True)
    parser.add_argument('--path', type=str, required=True, help='Path to image root directory')
    parser.add_argument('--data_path', type=str, default='data-bin', help='Path to save features')
    parser.add_argument('--box_num', type=int, default=36, help='Max number of regions per image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    model = FasterRCNN(backbone, num_classes=91).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    img_dir = os.path.join(args.path, image_dirs[args.dataset])
    txt_file = os.path.join(args.path, split_files[args.dataset])
    filenames = get_filenames(txt_file)

    save_dir = os.path.join(args.data_path, 'features_region', args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for file in tqdm(filenames):
            image_id = os.path.splitext(os.path.basename(file))[0]
            img_path = os.path.join(img_dir, os.path.basename(file))
            save_file = os.path.join(save_dir, f"{image_id}_region.npz")

            if os.path.exists(save_file):
                continue

            if not os.path.isfile(img_path):
                print(f"[Warning] Missing image: {img_path}")
                feats = torch.zeros((args.box_num, 2048))
                np.savez_compressed(save_file, feat=feats.numpy())
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = transform(image).to(device)
                x = image_tensor.unsqueeze(0)

                output = model(x)[0]
                boxes = output['boxes']
                topk = min(len(boxes), args.box_num)
                boxes = boxes[:topk]

                if boxes.shape[0] == 0:
                    feats = torch.zeros((args.box_num, 2048))
                else:
                    # --- 修改开始 ---
                    # 手动调用 backbone body 来获取未经 FPN 处理的原始 layer4 特征图
                    body = model.backbone.body
                    feat_map = body.conv1(x)
                    feat_map = body.bn1(feat_map)
                    feat_map = body.relu(feat_map)
                    feat_map = body.maxpool(feat_map)
                    feat_map = body.layer1(feat_map)
                    feat_map = body.layer2(feat_map)
                    feat_map = body.layer3(feat_map)
                    feat_map = body.layer4(feat_map) # 此时 feat_map 的通道数是 2048
                    # --- 修改结束 ---

                    box_indices = torch.zeros((boxes.size(0), 1), device=device)
                    rois = torch.cat([box_indices, boxes], dim=1)

                    roi_feats = roi_align(
                        feat_map,
                        rois,
                        output_size=(7, 7),
                        spatial_scale=1.0 / 32,
                        sampling_ratio=2,
                    )

                    roi_feats_pooled = torch.nn.functional.adaptive_avg_pool2d(
                        roi_feats, (1, 1)
                    ).squeeze(-1).squeeze(-1)


                    if roi_feats_pooled.size(0) < args.box_num:
                        pad = torch.zeros(args.box_num - roi_feats_pooled.size(0), 2048)
                        feats = torch.cat([roi_feats_pooled.cpu(), pad], dim=0)
                    else:
                        feats = roi_feats_pooled[:args.box_num].cpu()

                np.savez_compressed(save_file, feat=feats.numpy())

            except Exception as e:
                print(f"[Error] Failed to process {img_path}: {e}")
                feats = torch.zeros((args.box_num, 2048))
                np.savez_compressed(save_file, feat=feats.numpy())

    print(f"Region features saved to: {save_dir}")