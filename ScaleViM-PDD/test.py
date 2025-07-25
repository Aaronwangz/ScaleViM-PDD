import os
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
import yaml
from types import SimpleNamespace
from model006 import U_Net
from torch.utils.data import DataLoader
from dataset import myImageFlodertest
from metrics import ssim, psnr, lpips, niqe
import lpips as lpips_pkg

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 初始化设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dict_to_namespace(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)
    return SimpleNamespace(**d)


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return dict_to_namespace(config)


def compute_metrics(image1, image2, lpips_model):
    ssim_val = ssim(image1, image2)
    psnr_val = psnr(image1, image2)
    lpips_val = lpips(image1, image2, lpips_model)
    niqe_val = niqe(image1)
    return ssim_val, psnr_val, lpips_val, niqe_val


def save_reconstructed_image(reconstructed_image, idx, ssim_val, psnr_val, output_dir):
    """Save the reconstructed image with SSIM and PSNR in the filename."""
    os.makedirs(output_dir, exist_ok=True)

    # Create filename with SSIM and PSNR values
    reconstructed_filename = os.path.join(output_dir,
                                          f"{idx}_SSIM_{ssim_val:.5f}_PSNR_{psnr_val:.3f}.png")

    # Save the reconstructed image
    save_image(reconstructed_image, reconstructed_filename)


def test(dataset_path, output_dir='test_image_rsidpb1'):
    model = U_Net(dim=32).to(device)
    model.load_state_dict(torch.load('/opt/data/private/diffusemodel/manba/train_result_DHID/generator_epoch_76.pth', map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = myImageFlodertest(root=dataset_path, transform=transform, resize=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    total_ssim, total_psnr, total_lpips, total_niqe, count = 0, 0, 0, 0, 0

    # 初始化LPIPS模型
    lpips_model = lpips_pkg.LPIPS(net='alex').to(device)

    for idx, batch in enumerate(dataloader):
        blurry_image, clear_image = batch
        blurry_image, clear_image = blurry_image.to(device), clear_image.to(device)

        with torch.no_grad():
            reconstructed_image = model(blurry_image)

        ssim_val, psnr_val, lpips_val, niqe_val = compute_metrics(reconstructed_image, clear_image, lpips_model)
        total_ssim += ssim_val
        total_psnr += psnr_val
        total_lpips += lpips_val
        total_niqe += niqe_val

        # Save only the reconstructed image with SSIM and PSNR values in filenames
        save_reconstructed_image(reconstructed_image, idx, ssim_val, psnr_val, output_dir)

        count += 1

        print(f'Image {idx}')

    avg_ssim = total_ssim / count
    avg_psnr = total_psnr / count
    avg_lpips = total_lpips / count
    avg_niqe = total_niqe / count

    print(
        f'Average SSIM: {avg_ssim:.5f}, Average PSNR: {avg_psnr:.3f} dB, Average LPIPS: {avg_lpips:.5f}, Average NIQE: {avg_niqe:.5f}')


if __name__ == "__main__":
    # data_root = "../data/Haze1k/average/test_moderate"
    data_root = "../data/RSID/test"
    # data_root = "/opt/data/private/zq/data/RRSHID/thin_fog/test"
    test(data_root)


# import os
# import torch
# import torch.nn.functional as F
# import numpy as np
# import torchvision.transforms as transforms
# from torchvision.utils import save_image
# import yaml
# from types import SimpleNamespace
# from model006 import U_Net  # 确保 U_Net 从 model006 正确导入
# from torch.utils.data import DataLoader
# from dataset import myImageFlodertest
# from metrics import ssim, psnr, lpips, niqe
# import lpips as lpips_pkg
#
# # 导入 FLOPs 计算所需的库
# from thop import profile
# # 导入参数量美化输出的工具（可选）
# from thop.utils import clever_format
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# # 初始化设备
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# def dict_to_namespace(d):
#     for key, value in d.items():
#         if isinstance(value, dict):
#             d[key] = dict_to_namespace(value)
#     return SimpleNamespace(**d)
#
#
# def load_config(config_path):
#     with open(config_path, 'r', encoding='utf-8') as file:
#         config = yaml.safe_load(file)
#     return dict_to_namespace(config)
#
#
# def compute_metrics(image1, image2, lpips_model):
#     ssim_val = ssim(image1, image2)
#     psnr_val = psnr(image1, image2)
#     lpips_val = lpips(image1, image2, lpips_model)
#     niqe_val = niqe(image1)
#     return ssim_val, psnr_val, lpips_val, niqe_val
#
#
# def save_reconstructed_image(reconstructed_image, idx, ssim_val, psnr_val, output_dir):
#     """Save the reconstructed image with SSIM and PSNR in the filename."""
#     os.makedirs(output_dir, exist_ok=True)
#
#     # Create filename with SSIM and PSNR values
#     reconstructed_filename = os.path.join(output_dir,
#                                           f"{idx}_SSIM_{ssim_val:.5f}_PSNR_{psnr_val:.3f}.png")
#
#     # Save the reconstructed image
#     save_image(reconstructed_image, reconstructed_filename)
#
#
# def test(dataset_path, output_dir='test_image_RSID_L'):
#     model = U_Net(dim=32).to(device)
#     model.load_state_dict(torch.load('train_result_RSID_L3/prebest.pth', map_location=device))
#     model.eval()  # 将模型设置为评估模式，这对计算FLOPs很重要
#
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#
#     dataset = myImageFlodertest(root=dataset_path, transform=transform, resize=False)
#     # 计算FLOPs需要一个样本输入，通常批量大小设为1
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
#
#     # --- 计算参数量 ---
#     # 统计所有需要梯度的（即可训练的）参数
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"总可训练参数量: {total_params:,}")  # 格式化输出，用逗号分隔
#
#     # --- 计算 FLOPs ---
#     # 需要一个虚拟输入来计算 FLOPs。这里我们从数据集中取第一个批量作为示例输入。
#     # 确保输入形状与你的模型期望的输入形状一致，通常是 (batch_size, channels, height, width)。
#     # 根据你的参数，图片尺寸为 1200x1600，3通道。
#     # 我们将使用一个具有预期形状的虚拟输入。
#     input_dummy = torch.randn(1, 3, 256, 256).to(device)  # 批量大小1，3通道，1200x1600
#
#     # `profile` 函数会返回 MACs（乘加操作）和参数量。我们已经手动计算了参数量，所以主要关注 FLOPs。
#     # `custom_ops` 可用于 thop 不识别的自定义层，但对于标准层通常不需要。
#     # `verbose=False` 用于抑制详细的逐层输出。
#     macs, params = profile(model, inputs=(input_dummy,), verbose=False)
#
#     # 将 MACs（乘加操作）转换为 FLOPs。
#     # 一个常见的约定是 1 MAC = 2 FLOPs（一次乘法和一次加法）。
#     flops = macs * 2
#
#     # 可选：格式化 FLOPs 和参数量，使其更易读（例如，GFLOPs，M参数）
#     flops_formatted, params_formatted = clever_format([flops, total_params], "%.3f")
#
#     print(f"模型 FLOPs: {flops_formatted}")
#     print(f"模型参数量 (thop 统计): {params_formatted} (与手动统计一致: {total_params:,})")  # 显示两者以便比较
#
#     total_ssim, total_psnr, total_lpips, total_niqe, count = 0, 0, 0, 0, 0
#
#     # 初始化 LPIPS 模型
#     lpips_model = lpips_pkg.LPIPS(net='alex').to(device)
#
#     # 遍历 dataloader 进行实际测试
#     for idx, batch in enumerate(dataloader):
#         blurry_image, clear_image = batch
#         blurry_image, clear_image = blurry_image.to(device), clear_image.to(device)
#
#         with torch.no_grad():
#             reconstructed_image = model(blurry_image)
#
#         ssim_val, psnr_val, lpips_val, niqe_val = compute_metrics(reconstructed_image, clear_image, lpips_model)
#         total_ssim += ssim_val
#         total_psnr += psnr_val
#         total_lpips += lpips_val
#         total_niqe += niqe_val
#
#         # 只保存带有 SSIM 和 PSNR 值的重建图像
#         save_reconstructed_image(reconstructed_image, idx, ssim_val, psnr_val, output_dir)
#
#         count += 1
#
#         print(f'图像 {idx}')
#
#     avg_ssim = total_ssim / count
#     avg_psnr = total_psnr / count
#     avg_lpips = total_lpips / count
#     avg_niqe = total_niqe / count
#
#     print(
#         f'平均 SSIM: {avg_ssim:.5f}, 平均 PSNR: {avg_psnr:.3f} dB, 平均 LPIPS: {avg_lpips:.5f}, 平均 NIQE: {avg_niqe:.5f}')
#
#
# if __name__ == "__main__":
#     # data_root = "../data/Haze1k/average/test_moderate"
#     data_root = "../data/RSID/test"
#     test(data_root)
