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


# 加载ImageNet类别标签
def load_imagenet_labels():
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    response = requests.get(url)
    return response.text.strip().split('\n')


# 计算LRP注意力图
def compute_lrp_attention(image_path, target_class):
    weights = ResNet101_Weights.DEFAULT
    model = models.resnet101(weights=weights).eval()
    composite = EpsilonPlusFlat(canonizers=[ResNetCanonizer()])
    preprocess = weights.transforms()
    image = Image.open(image_path).convert('RGB')
    x = preprocess(image).unsqueeze(0)
    x.requires_grad_()

    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

    one_hot = torch.eye(logits.size(1), device=x.device)[[target_class]]
    with Gradient(model, composite) as attributor:
        _, relevance = attributor(x, one_hot)

    R = relevance.squeeze(0).sum(0).detach().cpu().numpy()
    R = np.maximum(R, 0)
    R /= (R.max() + 1e-12)
    R = R ** 0.5
    R = skimage.transform.resize(R, image.size[::-1])

    return image, R, probs, target_class


# 手动选择bounding boxes
def get_face_boxes_interactive(image_path):
    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    plt.title("Click two corners of each face box: top-left, bottom-right.\nClose window when done.")

    coords = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            ix, iy = int(event.xdata), int(event.ydata)
            print(f"Clicked at: ({ix}, {iy})")
            coords.append((ix, iy))

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    boxes = []
    for i in range(0, len(coords), 2):
        if i + 1 < len(coords):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            boxes.append(box)
    print(f"\nYou selected {len(boxes)} face boxes.")
    return boxes


# 计算每个人的attention得分
def compute_attention_per_person(attention_map, face_boxes):
    scores = []
    for box in face_boxes:
        x1, y1, x2, y2 = map(int, box)
        crop = attention_map[y1:y2, x1:x2]
        score = crop.mean()
        scores.append(score)
    return scores


# 可视化结果
def visualize_attention(image, face_boxes, scores):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    for i, (box, score) in enumerate(zip(face_boxes, scores)):
        x1, y1, x2, y2 = map(int, box)
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"Person {i + 1}: {score:.3f}",
                color='yellow', fontsize=12, backgroundcolor='black')
    max_idx = np.argmax(scores)
    ax.set_title(f'Person {max_idx + 1} gets the most attention!', fontsize=18)
    plt.axis('off')
    plt.show()


# 主函数
def main():
    image_path = "imgs/gr.png"

    # Step 1: 用ResNet预测类别
    weights = ResNet101_Weights.DEFAULT
    model = models.resnet101(weights=weights).eval()
    preprocess = weights.transforms()
    image = Image.open(image_path).convert("RGB")
    x = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
    top1_class = torch.argmax(probs).item()
    labels = load_imagenet_labels()
    print(f"Top predicted class: {labels[top1_class]} ({probs[top1_class] * 100:.2f}%)")

    # Step 2: LRP
    image, attention_map, probs, cls = compute_lrp_attention(image_path, top1_class)

    # Step 3: 手动框人脸
    face_boxes = get_face_boxes_interactive(image_path)

    # Step 4: 分数 + 显示
    scores = compute_attention_per_person(attention_map, face_boxes)
    visualize_attention(image, face_boxes, scores)


if __name__ == "__main__":
    main()
