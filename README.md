# 🎤 Smart Audio Deepfake Detection Platform  
### wav2vec 2.0 + RNN (Bi-LSTM) | Real-Time Voice Authenticity Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-wav2vec2-yellow.svg)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🧠 Overview

This repository contains a **highly engineered, end-to-end deep learning platform** for **audio deepfake detection**, designed to distinguish **real human speech** from **AI-generated or manipulated audio** with **high accuracy and real-time capability**.

The system leverages **state-of-the-art self-supervised learning (wav2vec 2.0)** combined with **recurrent neural networks (RNNs)** and **Bi-Directional LSTM (Bi-LSTM)** to model **temporal speech patterns** that are often imperfectly reproduced by synthetic voice generation systems.

This project is not a toy example — it demonstrates **real-world ML engineering**, including dataset construction, augmentation, transfer learning, temporal modeling, evaluation, and deployment readiness.

---

## 🚀 Why This Project Matters

With the explosion of:
- Voice cloning systems  
- AI-powered Text-to-Speech engines  
- Deepfake audio used in fraud, impersonation, and online exam malpractice  

There is a **critical need** for automated systems that can **verify the authenticity of speech**.

This platform directly addresses that need by focusing on:
- Robust feature extraction
- Temporal inconsistency detection
- Generalization to unseen fake audio
- Real-time inference feasibility

---

## ✨ Key Capabilities

- Deep speech representation learning using wav2vec 2.0
- Sequential modeling with RNN & Bi-Directional LSTM
- Intelligent dataset engineering from raw audio
- Class imbalance handling through realistic augmentation
- Binary classification: Real vs Fake
- 96% test accuracy on balanced evaluation set
- Real-time microphone-based prediction
- Lightweight, reusable trained model

---

## 🏗️ Complete System Architecture

Raw Audio (.wav, 16 kHz, mono)
↓
Segmentation into fixed 1-second chunks
↓
Data augmentation (applied only to real audio)
↓
wav2vec 2.0 pretrained encoder (frozen)
↓
High-level speech embeddings (49 × 768)
↓
Recurrent Neural Network (Bi-Directional LSTM)
↓
Fully Connected Dense Layer
↓
Sigmoid Activation
↓
Prediction: Real (0) or Fake (1)


Each stage of the pipeline is deliberately chosen to maximize robustness, interpretability, and performance.

---

## 🗂️ Dataset Engineering & Construction

### Practical Challenges Solved

- Raw datasets contained **very long recordings**
- Severe **class imbalance** between real and fake samples
- Limited availability of genuine human speech
- Hardware and storage constraints in cloud environments

This project addresses all of these issues through **careful dataset engineering**.

---

### 🔹 Audio Chunking Strategy

All recordings are split into **uniform 1-second segments**, ensuring:
- Fixed-length input for wav2vec
- Stable batch processing
- Consistent temporal resolution

```python
def chunk_audio(audio, sr, chunk_seconds=1):
    chunk_size = sr * chunk_seconds
    return [
        audio[i:i+chunk_size]
        for i in range(0, len(audio), chunk_size)
        if len(audio[i:i+chunk_size]) == chunk_size
    ]
 ```
### 🔹 Data Augmentation (Real Only)
```
def augment_audio(audio, sr):
    aug = []
    aug.append(audio)
    aug.append(audio + np.random.normal(0, 0.005, audio.shape))
    aug.append(librosa.effects.time_stretch(audio, rate=0.95))
    aug.append(librosa.effects.time_stretch(audio, rate=1.05))
    aug.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=1))
    return aug
```
| Split      | Real     | Fake     | Total    |
| ---------- | -------- | -------- | -------- |
| Train      | 800      | 800      | 1600     |
| Validation | 100      | 100      | 200      |
| Test       | 100      | 100      | 200      |
| **Total**  | **1000** | **1000** | **2000** |


📁 Dataset Structure
```
audio_dataset/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

🧠 Feature Extraction – wav2vec 2.0


```
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
wav2vec.eval()
```

Output shape

(49, 768)

🔁 Temporal Modeling – RNN (Bi-LSTM)

```
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return torch.sigmoid(self.fc(last))
```
⚙️ Training Configuration

Loss: Binary Cross Entropy

Optimizer: Adam

Learning Rate: 1e-3

Batch Size: 32

Epochs: 10

📈 Results
Test Accuracy: 96%

🎤 Real-Time Audio Inference
audio = record_audio(seconds=2)
embedding = extract_realtime_embedding(audio)
prediction = model(embedding)


Output

🟢 REAL AUDIO (Normal)
🔴 FAKE AUDIO DETECTED (Malpractice)

💾 Model Saving
torch.save(model.state_dict(), "bilstm_wav2vec_real_fake.pth")
```
models/
└── bilstm_wav2vec_real_fake.pth
```

## 🛠️ Tech Stack

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![wav2vec2](https://img.shields.io/badge/wav2vec%202.0-Speech%20Encoder-orange.svg)](https://huggingface.co/facebook/wav2vec2-base)
[![RNN](https://img.shields.io/badge/RNN-Sequence%20Modeling-purple.svg)](#)
[![BiLSTM](https://img.shields.io/badge/Bi--LSTM-Temporal%20Learning-indigo.svg)](#)
[![Librosa](https://img.shields.io/badge/librosa-Audio%20Processing-lightgrey.svg)](https://librosa.org/)
[![Torchaudio](https://img.shields.io/badge/torchaudio-Audio%20I/O-green.svg)](https://pytorch.org/audio/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML%20Tools-f7931e.svg)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blueviolet.svg)](https://numpy.org/)
[![Google%20Colab](https://img.shields.io/badge/Google%20Colab-Cloud%20Training-yellow.svg)](https://colab.research.google.com/)

🚀 Applications

 🔹Audio Deepfake Detection

 🔹Online Exam Malpractice Detection

 🔹Voice Authentication

 🔹Media Forensics

 🔹Cybersecurity & Fraud Prevention

🔮 Future Work

 🔹Fine-tune wav2vec encoder

 🔹Attention-based RNN models

 🔹Transformer temporal modeling

 🔹Multimodal deepfake detection

 🔹Web / Mobile deployment

📌 Note

Datasets are not included due to size constraints.
All steps are fully reproducible using the provided notebook.

👨‍💻 Author

Built as a complete end-to-end deep learning project using
wav2vec 2.0 + RNN (Bi-LSTM) for secure and trustworthy AI audio systems.
