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

class MultiScaleCNN(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=128):
        super(MultiScaleCNN, self).__init__()
        
        # Уровень 1: Детальные признаки (низкоуровневые)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # ↓ пространственное разрешение, ↑ семантическая плотность
        )
        
        # Уровень 2: Признаки среднего уровня
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.MaxPool2d(2)  # ↓ пространственное разрешение, ↑ семантическая плотность
        )
        
        # Уровень 3: Высокоуровневые признаки
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # ↓ до фиксированного размера
        )
        
        # Глобальный пулинг для каждого уровня (альтернатива выравниванию)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Объединение РАЗНЫХ ТИПОВ признаков
        total_features = 64 + 128 + 256  # Сумма каналов со всех уровней
        self.fc = nn.Linear(total_features, embedding_dim)
        
    def forward(self, x):
        # Извлекаем признаки разного уровня
        features1 = self.conv1(x)  # Низкоуровневые
        features2 = self.conv2(features1)  # Среднеуровневые  
        features3 = self.conv3(features2)  # Высокоуровневые
        
        # Применяем глобальный пулинг к каждому уровню
        global1 = self.global_pool(features1)  # [batch, 64, 1, 1]
        global2 = self.global_pool(features2)  # [batch, 128, 1, 1]
        global3 = self.global_pool(features3)  # [batch, 256, 1, 1]
        
        # Выравниваем и объединяем РАЗНЫЕ ТИПЫ признаков
        global1_flat = global1.view(global1.size(0), -1)  # [batch, 64]
        global2_flat = global2.view(global2.size(0), -1)  # [batch, 128]
        global3_flat = global3.view(global3.size(0), -1)  # [batch, 256]
        
        # ОБЪЕДИНЕНИЕ: соединяем разные семантические уровни
        combined_semantic = torch.cat([global1_flat, global2_flat, global3_flat], dim=1)
        
        # Финальный эмбеддинг
        embedding = self.fc(combined_semantic)
        
        return embedding

class TextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, lstm_hidden=512, 
                 hidden_dims=[1024, 512, 512], final_dim=512):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden, batch_first=True, bidirectional=True)
        
        # Динамическое создание линейных слоев
        layers = []
        input_size = lstm_hidden * 2  # bidirectional
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_dim
        
        # Финальный слой
        layers.append(nn.Linear(input_size, final_dim))
        
        self.linear_layers = nn.Sequential(*layers)
        self.embedding_size = final_dim
    
    def forward(self, input_ids):
        # Эмбеддинг + LSTM
        x = self.embedding(input_ids)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        
        # Через линейные слои
        embeddings = self.linear_layers(last_hidden)
        return embeddings

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
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
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