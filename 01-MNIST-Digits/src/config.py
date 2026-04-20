"""
Configurazione Centrale per il Modulo MNIST
-------------------------------------------
Questo file funge da "pannello di controllo" per l'intero progetto.
Modificando i parametri qui, cambierà il comportamento di training
e inferenza in tutti gli altri script.
"""

import torch
from pathlib import Path

# ==========================================
# 1. GESTIONE DEI PERCORSI (PATHS)
# ==========================================
# Usiamo pathlib.Path(__file__).parent.parent per risalire dinamicamente 
# alla cartella principale "01-MNIST-Digits", a prescindere da dove avvii lo script!
BASE_DIR = Path(__file__).resolve().parent.parent

# Definiamo le sottocartelle usando l'operatore "/" fornito da pathlib
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Assicuriamoci che le cartelle esistano (le crea se mancano, ignorando se già ci sono)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. IPERPARAMETRI DEL MODELLO (HYPERPARAMETERS)
# ==========================================
# Questi sono le "manopole" che giriamo per far imparare meglio la rete neurale
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 5

# Dimensioni dell'immagine MNIST (Scala di grigi, 28x28 pixel)
IMAGE_CHANNELS = 1
IMAGE_SIZE = 28

# Valori standard matematici per normalizzare i tensori MNIST
# (Media e Deviazione Standard calcolate sull'intero dataset globale)
NORM_MEAN = 0.1307
NORM_STD = 0.3081


# ==========================================
# 3. CONFIGURAZIONE DI SISTEMA E HARDWARE
# ==========================================
# Rilevamento intelligente della GPU: usa CUDA (NVIDIA) se disponibile, altrimenti CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Nome del file in cui verrà salvato il modello addestrato
MODEL_SAVE_PATH = MODELS_DIR / "mnist_classifier_v1.pt"