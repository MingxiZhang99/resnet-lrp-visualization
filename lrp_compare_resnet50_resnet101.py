import torch
import numpy as np
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights
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


def lrp_model(img_path, model_name, class_indices=None):
    if model_name == 'ResNet50':
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights).eval()
        composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
    elif model_name == 'ResNet101':
        weights = ResNet101_Weights.DEFAULT
        model = models.resnet101(weights=weights).eval()
        composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
    else:
        raise ValueError("Model must be 'ResNet50' or 'ResNet101'")

    preprocess = weights.transforms()
    image = Image.open(img_path).convert('RGB')
    x = preprocess(image).unsqueeze(0)
    x.requires_grad_()

    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

    labels = load_imagenet_labels()

    if class_indices is None:
        topk_probs, topk_classes = torch.topk(probs, 4)
        class_indices = topk_classes.tolist()

    attention_maps = []
    for cls in class_indices:
        one_hot = torch.eye(logits.size(1), device=x.device)[[cls]]
        with Gradient(model, composite) as attributor:
            _, relevance = attributor(x, one_hot)
        R = relevance.squeeze(0).sum(0).detach().cpu().numpy()
        R = np.maximum(R, 0)
        R /= (R.max() + 1e-12)
        R = R ** 0.5  # 非线性增强亮度
        R = skimage.transform.resize(R, image.size[::-1])

        attention_maps.append(R)

    return image, model_name, [probs[cls].item() for cls in class_indices], class_indices, attention_maps


def plot_attention_maps(image, model_name, probs, class_indices, attention_maps, global_max, overlay_on_image=True):
    labels = load_imagenet_labels()
    plt.figure(figsize=(5 * (len(class_indices) + 1), 5))
    plt.subplot(1, len(class_indices) + 1, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    for i, (cls, R) in enumerate(zip(class_indices, attention_maps)):
        plt.subplot(1, len(class_indices) + 1, i + 2)
        if overlay_on_image:
            plt.imshow(image)
            plt.imshow(R, alpha=0.7, cmap='jet', vmin=0, vmax=global_max)
        else:
            plt.imshow(R, cmap='jet', vmin=0, vmax=global_max)
        plt.title(f"{labels[cls]}: {probs[i]*100:.2f}%")
        plt.axis('off')

    plt.suptitle(f"Model: {model_name}", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_combined_attention_maps(results, global_max, overlay_on_image=True):
    labels = load_imagenet_labels()

    num_classes = len(results[0][3])  # class_indices 长度
    num_models = len(results)

    # 创建子图
    fig, axs = plt.subplots(num_models, num_classes + 1, figsize=(5 * (num_classes + 1), 5 * num_models))

    if num_models == 1:
        axs = [axs]  # 保证可以使用双下标 axs[row][col]

    # 遍历每个模型的结果
    for row, (image, model_name, probs, class_indices, attention_maps) in enumerate(results):
        # 标注原图并显示
        axs[row][0].imshow(image)
        axs[row][0].set_title(f"Original Image\n({model_name})", fontsize=12)
        axs[row][0].axis('off')

        # 遍历每个类别的 attention map
        for col, (cls, R) in enumerate(zip(class_indices, attention_maps), start=1):
            if overlay_on_image:
                axs[row][col].imshow(image)
                axs[row][col].imshow(R, alpha=0.7, cmap='jet', vmin=0, vmax=global_max)
            else:
                axs[row][col].imshow(R, cmap='jet', vmin=0, vmax=global_max)

            # 添加标题和图例
            axs[row][col].set_title(f"{labels[cls]}: {probs[col-1]*100:.2f}%", fontsize=10)
            axs[row][col].axis('off')

        # 标注模型名
        axs[row][0].set_ylabel(f"{model_name}", fontsize=14, rotation=0, labelpad=40, loc='center')

    # 调整布局，使图形不重叠
    plt.tight_layout()
    plt.show()

def main():
    img_path = "img/n02206856_bee.JPEG"
    # img_path = "img/n01491361_tiger_shark.JPEG"
    # img_path = "img/n12998815_agaric.JPEG"

    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights).eval()
    preprocess = weights.transforms()
    image = Image.open(img_path).convert("RGB")
    x = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
    top5_classes = torch.topk(probs, 5).indices.tolist()

    predicted_class = top5_classes[0]
    similar_classes = top5_classes[1:3]
    dissimilar_class = 666

    selected_classes = [predicted_class] + similar_classes + [dissimilar_class]

    results = []
    all_max_values = []

    for model_name in ['ResNet50', 'ResNet101']:
        print(f"Processing {model_name}...")
        image, name, probs, cls_ids, attn_maps = lrp_model(
            img_path, model_name=model_name, class_indices=selected_classes)
        results.append((image, name, probs, cls_ids, attn_maps))
        all_max_values += [R.max() for R in attn_maps]

    global_max = max(all_max_values)

    for image, name, probs, cls_ids, attn_maps in results:
        # plot_attention_maps(image, name, probs, cls_ids, attn_maps,
        #                     global_max=global_max, overlay_on_image=True)
        plot_combined_attention_maps(results, global_max=global_max, overlay_on_image=True)



if __name__ == "__main__":
    main()
