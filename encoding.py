from sentence_transformers import SentenceTransformer
from params import *
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import normalize
import librosa
import numpy as np
import torch.nn.functional as F

class MultiScaleCNN(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=768):
        super(MultiScaleCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))  # Уменьшаем до 2x2
        )
        
        # После пулинга: [B, 64, 2, 2] → 64*2*2 = 256
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # [B, 256]
        embedding = self.fc(x)
        return F.normalize(embedding, p=2, dim=1)

class TextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=768):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False  # Упрощаем: односторонний, чтобы не удваивать размер
        )
        self.embedding_size = hidden_dim

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [B, T, D_emb]
        output, hidden = self.gru(x)   # output: [B, T, H], hidden: [1, B, H]
        
        # Берем последнее состояние
        last_hidden = output[:, -1, :]  # [B, H]
        
        # Нормализуем
        normalized = F.normalize(last_hidden, p=2, dim=1)
        return normalized

class TextEncoder():
    def __init__(self, model_name="thenlper/gte-base"):
        self.model = SentenceTransformer(model_name).to(DEVICE)

    def encode_dict(self, dct: dict):
        sentences = list(dct.values())
        ids = list(dct.keys())

        text_embeddings = self.model.encode(sentences, batch_size=TEXT_ENCODE_BATCH_SIZE, show_progress_bar=True)
        embeddings_normalized = normalize(text_embeddings, norm='l2')

        res = {i: v for i, v in zip(ids, embeddings_normalized)}

        return res

class AudioEncoder():
    def __init__(self, model=models.resnet18(weights='IMAGENET1K_V1')):
        self.model = model
        self.model.fc = nn.Identity()
        self.model = self.model.to(DEVICE)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

    def wav_to_spectrogram_3ch(self, wav_path, n_fft=2048, hop_length=512, n_mels=224):

        y, sr = librosa.load(wav_path, sr=None)

        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

        magnitude_spectrogram = np.abs(stft)

        log_spectrogram = librosa.amplitude_to_db(magnitude_spectrogram, ref=1.0)

        spec_min = log_spectrogram.min()
        spec_max = log_spectrogram.max()

        if spec_max == spec_min:
            normalized_spectrogram = np.zeros_like(log_spectrogram, dtype=np.float64)
        else:
            normalized_spectrogram = (log_spectrogram - spec_min) / (spec_max - spec_min)
            normalized_spectrogram = np.clip(normalized_spectrogram, 0.0, 1.0)

        spec_255 = (normalized_spectrogram * 255).astype(np.uint8)

        pil_image_obj = Image.fromarray(spec_255, mode='L')

        pil_image_resized = pil_image_obj.resize((224, 224), resample=Image.Resampling.LANCZOS)

        spec_array_resized = np.array(pil_image_resized)
        spec_3ch_array = np.stack([spec_array_resized] * 3, axis=-1)

        pil_image_3ch = Image.fromarray(spec_3ch_array, mode='RGB')

        return pil_image_3ch


    def encode_from_paths(self, dict_audio_paths) -> dict:
        audio_embeddings = {}
        audio_items = list(dict_audio_paths.items()) 
        total_items = len(audio_items)

        with torch.no_grad():
            for i in tqdm(range(0, total_items, IMAGE_ENCODE_BATCH_SIZE)):
                batch_items = audio_items[i:i + IMAGE_ENCODE_BATCH_SIZE]
                batch_paths = [audio_path for g_id, audio_path in batch_items] 
                batch_g_ids = [g_id for g_id, audio_path in batch_items]

                batch_tensors = []
                for audio_path in batch_paths: 
                    
                    img_pil = self.wav_to_spectrogram_3ch(audio_path)
                    
                    img_tensor = self.transform(img_pil)
                    batch_tensors.append(img_tensor)

                batch_img_tensor = torch.stack(batch_tensors).to(DEVICE)

                batch_embeddings = self.model(batch_img_tensor)

                for g_id, embedding in zip(batch_g_ids, batch_embeddings):
                    audio_embeddings[g_id] = embedding.cpu()

        return audio_embeddings 

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

    def encode_from_paths(self, dict_images_paths) -> dict:
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