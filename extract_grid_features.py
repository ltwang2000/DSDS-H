import os
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import timm
import torch.nn.functional as F
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

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
    parser.add_argument('--path', type=str, required=True, help='Image + txt 根目录（包含 *.txt 和 images 文件夹）')
    parser.add_argument('--data_path', type=str, default='data-bin', help='输出特征保存目录')
    parser.add_argument('--model', type=str, default='vit_base_patch16_224', help='ViT 模型名称')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Extracting ViT {args.model} grid features for {args.dataset}")
    model = timm.create_model(args.model, pretrained=True, num_classes=0, global_pool='').to(device)
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    img_dir = os.path.join(args.path, image_dirs[args.dataset])
    txt_file = os.path.join(args.path, split_files[args.dataset])
    filenames = get_filenames(txt_file)

    save_dir = os.path.join(args.data_path, 'features_grid', args.dataset)
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for file in tqdm(filenames):
            image_id = os.path.splitext(os.path.basename(file))[0]
            img_path = os.path.join(img_dir, os.path.basename(file))
            save_file = os.path.join(save_dir, f"{image_id}_grid.pth")

            if os.path.exists(save_file):
                continue
            if not os.path.isfile(img_path):
                print(f"[Warning] Missing image: {img_path}")
                feat = torch.zeros((196, 768))  # 补零避免中断
                torch.save({'feat': feat}, save_file)
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0).to(device)
                output = model.forward_features(input_tensor)

                if output.dim() == 3 and output.shape[1] > 1:
                    patch_tokens = output[:, 1:, :]  # 去除 CLS token，shape=(1, N, 768)
                    feat = patch_tokens.squeeze(0)   # (N, 768)

                    if feat.shape[0] != 196:
                        # 自适应池化固定为196个patch
                        feat = feat.transpose(0, 1).unsqueeze(0)  # (1, 768, N)
                        feat = F.adaptive_avg_pool1d(feat, 196)  # (1, 768, 196)
                        feat = feat.squeeze(0).transpose(0, 1)   # (196, 768)
                else:
                    print(f"[Failed] {img_path}: Invalid shape {output.shape}")
                    feat = torch.zeros((196, 768))

                torch.save({'feat': feat.cpu()}, save_file)

            except Exception as e:
                print(f"[Error] Failed to process {img_path}: {e}")
                feat = torch.zeros((196, 768))
                torch.save({'feat': feat}, save_file)

    print(f"Grid features saved to: {save_dir}")
