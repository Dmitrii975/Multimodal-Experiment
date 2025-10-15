from sentence_transformers import SentenceTransformer
from params import *
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import normalize


class TextEncoder():
    def __init__(self, model_name="thenlper/gte-base"):
        self.model = SentenceTransformer(model_name).to(DEVICE)
    def encode_list(self, sentences):
        text_embeddings = self.model.encode(sentences, batch_size=TEXT_ENCODE_BATCH_SIZE, show_progress_bar=True)
        embeddings_normalized = normalize(text_embeddings, norm='l2')

        return embeddings_normalized

class ImageEncoder():
    def __init__(self, model=models.resnet18(weights='IMAGENET1K_V1')):

        self.model = model
        self.model.fc = nn.Identity()
        self.model = self.model.to(DEVICE)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def encode_from_paths(self, dict_images_paths):
        img_embeddings = {}
        image_items = list(dict_images_paths.items())
        total_items = len(image_items)

        with torch.no_grad():
            for i in tqdm(range(0, total_items, IMAGE_ENCODE_BATCH_SIZE)):
                batch_items = image_items[i:i + IMAGE_ENCODE_BATCH_SIZE]
                batch_paths = [img_path for g_id, img_path in batch_items]
                batch_g_ids = [g_id for g_id, img_path in batch_items]

                batch_tensors = []
                for img_path in batch_paths:
                    with Image.open(img_path).convert('RGB') as img:
                        img_tensor = self.transform(img)
                        batch_tensors.append(img_tensor)

                batch_img_tensor = torch.stack(batch_tensors).to(DEVICE)

                batch_embeddings = self.model(batch_img_tensor)

                for g_id, embedding in zip(batch_g_ids, batch_embeddings):
                    img_embeddings[g_id] = embedding.cpu()

        return img_embeddings