from pathlib import Path
import torch

# --- GESTIONE PERCORSI (Agnosticismo OS) ---
# Risale di un livello da src/ per arrivare alla root del modulo 02
BASE_DIR = Path(__file__).resolve().parent.parent

# Percorsi principali (fuori da src/ per evitare confusione)

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = OUTPUT_DIR / "logs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions" # Nuova cartella per le immagini

# Sottocartelle specifiche
TEST_IMAGES_DIR = BASE_DIR / "test" / "img"
MODEL_PATH = MODELS_DIR / "modello_cifar10.pt"
LOG_CSV_PATH = LOGS_DIR / "classificazioni_log.csv"
HISTORY_FILE = LOGS_DIR / ".prediction_history.txt"

# Creazione automatica delle directory
for path in [DATA_DIR, MODELS_DIR, LOGS_DIR, PREDICTIONS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# --- IPERPARAMETRI DI TRAINING ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 20  # CIFAR-10 richiede più epoche di MNIST

# --- SPECIFICHE DATASET (CIFAR-10 RGB) ---
# Normalizzazione standard calcolata sul dataset CIFAR-10
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2023, 0.1994, 0.2010)

CLASSES = (
    'aereo', 'auto', 'uccello', 'gatto', 'cervo',
    'cane', 'rana', 'cavallo', 'nave', 'camion'
)
