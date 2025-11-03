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


class TextEncoder():
    def __init__(self, model_name="thenlper/gte-base"):
        self.model = SentenceTransformer(model_name).to(DEVICE)
    def encode_list(self, sentences):
        text_embeddings = self.model.encode(sentences, batch_size=TEXT_ENCODE_BATCH_SIZE, show_progress_bar=True)
        embeddings_normalized = normalize(text_embeddings, norm='l2')

        return embeddings_normalized

class AudioEncoder():
    def init(self, model=models.resnet18(weights='IMAGENET1K_V1')):

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
        # Load the audio file
        y, sr = librosa.load(wav_path, sr=None)  # Keep original sample rate

        # Compute the Short-Time Fourier Transform (STFT)
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

        # Compute the magnitude spectrogram
        magnitude_spectrogram = np.abs(stft)

        # Convert to dB scale (log-magnitude)
        log_spectrogram = librosa.amplitude_to_db(magnitude_spectrogram, ref=np.max)

        # Resize the spectrogram to 224x224 using PIL for resampling
        # Transpose to (time, freq) for PIL, then resize
        spec_resized = np.array(Image.fromarray(log_spectrogram).resize((224, 224), resample=Image.BICUBIC))

        # Normalize the spectrogram to [0, 1] range
        spec_normalized = (spec_resized - spec_resized.min()) / (spec_resized.max() - spec_resized.min())

        # Repeat the single channel to create 3 channels (like an RGB image)
        spec_3ch = np.stack([spec_normalized] * 3, axis=-1)  # Shape: (224, 224, 3)

        pil_image = Image.fromarray((spec_3ch * 255).astype(np.uint8), mode='RGB')

        return pil_image

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
                    img = self.wav_to_spectrogram_3ch(img_path)
                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)

                batch_img_tensor = torch.stack(batch_tensors).to(DEVICE)

                batch_embeddings = self.model(batch_img_tensor)

                for g_id, embedding in zip(batch_g_ids, batch_embeddings):
                    img_embeddings[g_id] = embedding.cpu()

        return img_embeddings

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