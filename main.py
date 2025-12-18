import os
from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

# 载入模型
model_name = "prithivMLmods/Trash-Net"
processor = AutoImageProcessor.from_pretrained(model_name)
model = SiglipForImageClassification.from_pretrained(model_name, trust_remote_code=True)
model.eval()

# 图片文件夹路径
img_dir = "images"

# 获取所有文件名
image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

results = []

# 批量处理
for filename in image_files:
    path = os.path.join(img_dir, filename)
    try:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

        labels = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
        predictions = {labels[i]: round(probs[i], 3) for i in range(len(probs))}

        trash_score = predictions["trash"]
        results.append((filename, trash_score, predictions))
        print(f"✅ {filename:25s}  Trash Index: {trash_score:.3f}")

    except Exception as e:
        print(f"⚠️ Error processing {filename}: {e}")

# 按“trash 概率”排序
results.sort(key=lambda x: x[1], reverse=True)

print("\n=== Future Discard Index Ranking ===")
for rank, (fname, score, preds) in enumerate(results, 1):
    print(f"{rank:2d}. {fname:25s}  Trash Index: {score:.3f}")
