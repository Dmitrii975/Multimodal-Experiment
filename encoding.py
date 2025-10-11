from params import *
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import glob
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
    def __init__(self, model):

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

    def encode_from_path(self, images_path):
        self.image_files = glob.glob(images_path + '/*.jpg')
        img_embeddings = dict()

        with torch.no_grad():
            for img_path in tqdm(self.image_files):
                with Image.open(img_path).convert('RGB') as img:
                    img_tensor = self.transform(img).unsqueeze(0).to(DEVICE)

                    embedding = self.model(img_tensor)
                    img_embeddings[img_path.split('/')[-1]] = embedding.cpu()

        return img_embeddings