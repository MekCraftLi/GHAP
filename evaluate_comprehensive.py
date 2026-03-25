import os
import json
import math
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np

# 导入 3DGS 原版指标计算函数
from utils.loss_utils import ssim
from utils.image_utils import psnr
from lpipsPyTorch import lpips


def calculate_masked_psnr(render, gt, mask):
    """
    计算基于掩码的前景 PSNR。
    render, gt: [1, 3, H, W]
    mask: [1, 1, H, W] boolean
    """
    mask_squeeze = mask.squeeze()
    if mask_squeeze.sum() == 0:
        return 0.0  # 掩码全空的情况

    render_fg = render[0, :, mask_squeeze]  # [3, N]
    gt_fg = gt[0, :, mask_squeeze]  # [3, N]

    mse = torch.mean((render_fg - gt_fg) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0) - 10 * math.log10(mse.item())


def readImagesAndMasks(renders_dir, gt_dir, mask_dir):
    renders = []
    gts = []
    masks = []
    image_names = []

    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)

        # 加载对应的掩码
        mask_path = mask_dir / (fname.split('.')[0] + ".png")
        if mask_path.exists():
            mask_np = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            mask_t = torch.from_numpy(mask_np).to("cuda") > 127
            masks.append(mask_t.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]
        else:
            masks.append(None)

    return renders, gts, masks, image_names


def evaluate_comprehensive(model_paths, source_path):
    print("=== 开始综合质量评估 (Global & Foreground) ===")
    mask_dir = Path(source_path) / "edge_masks"
    if not mask_dir.exists():
        print(f"[警告] 未找到掩码目录: {mask_dir}，只能进行全局评估！")

    full_dict = {}

    for scene_dir in model_paths:
        try:
            print(f"\n[评估场景]: {scene_dir}")
            full_dict[scene_dir] = {}
            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print(f"[{method}] 读取图像中...")
                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"

                renders, gts, masks, image_names = readImagesAndMasks(renders_dir, gt_dir, mask_dir)

                global_psnrs, global_ssims, global_lpipss = [], [], []
                fg_psnrs, fg_ssims, fg_lpipss = [], [], []

                for idx in tqdm(range(len(renders)), desc="计算指标进度"):
                    render, gt, mask = renders[idx], gts[idx], masks[idx]

                    # 1. 计算全局指标
                    global_psnrs.append(psnr(render, gt).item())
                    global_ssims.append(ssim(render, gt).item())
                    global_lpipss.append(lpips(render, gt, net_type='vgg').item())

                    # 2. 计算前景掩码指标
                    if mask is not None:
                        # Masked PSNR (精确的像素级前景误差)
                        fg_psnrs.append(calculate_masked_psnr(render, gt, mask))

                        # Masked SSIM & LPIPS (通过屏蔽背景黑块来计算结构)
                        # 将背景涂黑，让网络只关注主体的结构相似度
                        render_fg_only = render * mask
                        gt_fg_only = gt * mask

                        fg_ssims.append(ssim(render_fg_only, gt_fg_only).item())
                        fg_lpipss.append(lpips(render_fg_only, gt_fg_only, net_type='vgg').item())

                # 计算平均值
                res = {
                    "Global": {
                        "PSNR": np.mean(global_psnrs),
                        "SSIM": np.mean(global_ssims),
                        "LPIPS": np.mean(global_lpipss)
                    },
                    "Foreground": {
                        "PSNR": np.mean(fg_psnrs) if fg_psnrs else 0.0,
                        "SSIM": np.mean(fg_ssims) if fg_ssims else 0.0,
                        "LPIPS": np.mean(fg_lpipss) if fg_lpipss else 0.0
                    }
                }

                full_dict[scene_dir][method] = res

                print("\n================ 评估报告 ================")
                print(f"{'Metric':<15} | {'Global (全局)':<15} | {'Foreground (前景主体)':<15}")
                print("-" * 50)
                print(f"{'PSNR (↑)':<15} | {res['Global']['PSNR']:<15.4f} | {res['Foreground']['PSNR']:<15.4f}")
                print(f"{'SSIM (↑)':<15} | {res['Global']['SSIM']:<15.4f} | {res['Foreground']['SSIM']:<15.4f}")
                print(f"{'LPIPS(↓)':<15} | {res['Global']['LPIPS']:<15.4f} | {res['Foreground']['LPIPS']:<15.4f}")
                print("==========================================")

            # 保存到综合报告 json
            with open(scene_dir + "/comprehensive_results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=4)
            print(f"详细结果已保存至: {scene_dir}/comprehensive_results.json")

        except Exception as e:
            print(f"无法完成评估 {scene_dir}: {e}")


if __name__ == "__main__":
    parser = ArgumentParser(description="综合指标评估脚本")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, help="模型输出路径")
    parser.add_argument('--source_path', '-s', required=True, type=str, help="数据集原始路径(为了读取edge_masks)")
    args = parser.parse_args()

    evaluate_comprehensive(args.model_paths, args.source_path)