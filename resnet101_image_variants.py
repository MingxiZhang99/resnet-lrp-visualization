import torch
import numpy as np
from torchvision import models
from torchvision.models import ResNet101_Weights
from PIL import Image
import matplotlib.pyplot as plt
import skimage.transform
import requests

from zennit.torchvision import ResNetCanonizer
from zennit.composites import EpsilonPlusFlat
from zennit.attribution import Gradient


def load_imagenet_labels():
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    response = requests.get(url)
    return response.text.strip().split('\n')


def lrp_model(img_path, class_indices):
    weights = ResNet101_Weights.DEFAULT
    model = models.resnet101(weights=weights).eval()
    composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])

    preprocess = weights.transforms()
    image = Image.open(img_path).convert('RGB')
    x = preprocess(image).unsqueeze(0)
    x.requires_grad_()

    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

    labels = load_imagenet_labels()

    attention_maps = []
    for cls in class_indices:
        one_hot = torch.eye(logits.size(1), device=x.device)[[cls]]
        with Gradient(model, composite) as attributor:
            _, relevance = attributor(x, one_hot)
        R = relevance.squeeze(0).sum(0).detach().cpu().numpy()
        R = np.maximum(R, 0)
        R /= (R.max() + 1e-12)
        R = R ** 0.5
        R = skimage.transform.resize(R, image.size[::-1])
        attention_maps.append(R)

    return image, probs, class_indices, attention_maps


def plot_combined_attention_maps_grid(results, global_max, overlay_on_image=True):
    labels = load_imagenet_labels()

    # 定义标题的顺序
    row_titles = ["Original", "Rotation", "RGB flip", "Text"]
    col_titles = ["bee", "hen", "chameleon"]

    num_rows = 4  # 四行
    num_cols = 3  # 三列

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 12))

    # 关闭所有坐标轴
    for r in range(num_rows):
        for c in range(num_cols):
            axs[r][c].axis('off')

    # 放置图像并添加标题
    idx = 0
    for i in range(num_rows):  # 行
        for j in range(num_cols):  # 列
            if idx >= len(results):
                break
            image, probs, class_indices, attention_maps = results[idx]
            ax = axs[i][j]
            ax.imshow(image)
            if len(attention_maps) >= 1:
                R = attention_maps[0]
                cls = class_indices[0]
                if overlay_on_image:
                    ax.imshow(R, alpha=0.5, cmap='jet', vmin=0, vmax=global_max)
                # 修改为在标题中显示预测类别和概率
                ax.set_title(f"{row_titles[i]}_{col_titles[j]} score: {probs[cls]*100:.1f}%", fontsize=12)
            idx += 1

    # 调整布局以适应标题和图像
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()



def main():
    img_paths = [
        "imgs/b.jpg", "imgs/c.jpg", "imgs/s.jpg", "imgs/br.png", "imgs/cr.png", "imgs/sr.png","imgs/brf.png", "imgs/crf.png","imgs/srf.png","imgs/bt.jpg","imgs/ct.jpg", "imgs/st.jpg"
    ]

    weights = ResNet101_Weights.DEFAULT
    model = models.resnet101(weights=weights).eval()
    preprocess = weights.transforms()

    results = []
    all_max_values = []

    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        x = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
        top1_classes = torch.topk(probs, 1).indices.tolist()
        selected_classes = top1_classes

        print(f"Processing {img_path}...")
        image, probs, cls_ids, attn_maps = lrp_model(img_path, selected_classes)
        results.append((image, probs, cls_ids, attn_maps))
        all_max_values += [R.max() for R in attn_maps]

    global_max = max(all_max_values)
    # 不再分组，直接画一张完整的图
    plot_combined_attention_maps_grid(results, global_max=global_max, overlay_on_image=True)


if __name__ == "__main__":
    main()
