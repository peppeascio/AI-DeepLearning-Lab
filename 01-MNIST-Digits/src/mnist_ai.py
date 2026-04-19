import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets, transforms
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. IMPORTIAMO IL NOSTRO PANNELLO DI CONTROLLO!
import config

# --- DEFINIZIONE MODELLO ---
class DigitNet(nn.Module):
    """
    Rete Neurale Convoluzionale (CNN) per la classificazione di cifre (MNIST).
    Architettura: 2 strati Convoluzionali + Max Pooling -> Flatten -> 2 strati Lineari.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(config.IMAGE_CHANNELS, 32, 3), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# --- FUNZIONE DI TRAINING ---
# Aggiunti i Type Hints: lr è float, batch_size è int, epochs è int. Ritorna un dizionario (dict).
def run_experiment(lr: float, batch_size: int, epochs: int) -> dict:
    """
    Esegue un singolo esperimento di addestramento e validazione.
    Ritorna uno storico con i valori di loss e accuratezza per ogni epoca.
    """
    experiment_name = f"LR_{lr}_BS_{batch_size}"
    
    # Usiamo i percorsi definiti nel config (pathlib usa l'operatore /)
    log_dir = config.OUTPUT_DIR / experiment_name
    
    try:
        writer = SummaryWriter(log_dir=str(log_dir))
    except Exception as e:
        print(f"⚠️ Errore TensorBoard: {e}")
        writer = None
    
    print(f"\n🚀 Avvio esperimento: {experiment_name}")
    
    # Caricamento Dati usando i valori del config.py
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((config.NORM_MEAN,), (config.NORM_STD,))
    ])
    
    train_set = datasets.MNIST(root=config.DATA_DIR, train=True, download=True, transform=transform)
    val_set = datasets.MNIST(root=config.DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = DigitNet().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'loss': [], 'acc': []}

    for epoca in range(epochs):
        model.train()
        for data, target in tqdm(train_loader, desc=f"Epoca {epoca+1}/{epochs}"):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0, 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                out = model(data)
                v_loss += criterion(out, target).item()
                _, pred = out.max(1)
                v_total += target.size(0)
                v_correct += pred.eq(target).sum().item()

        epoch_loss = v_loss / len(val_loader)
        epoch_acc = 100. * v_correct / v_total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        
        if writer:
            writer.add_scalar('Loss/Validation', epoch_loss, epoca)
            writer.add_scalar('Accuracy/Validation', epoch_acc, epoca)
        
    if writer: 
        writer.close()
        
    # Salva il modello finale addestrato (solo l'ultimo, o potremmo salvare il migliore)
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    
    return history


# --- ESECUZIONE PRINCIPALE DELLO SCRIPT ---
# Questa è la "Guardia"! Tutto ciò che è qui dentro gira solo se esegui questo file direttamente.
if __name__ == "__main__":
    
    # Pulizia automatica della cartella di output prima di iniziare nuovi test
    if config.OUTPUT_DIR.exists():
        print(f"🧹 Pulizia in corso: rimozione vecchi log in {config.OUTPUT_DIR}...")
        shutil.rmtree(config.OUTPUT_DIR)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parametri per l'esperimento (potremmo prendere anche questi dal config se volessimo)
    learning_rates = [0.01, 0.001, 0.0001]
    all_results = {}

    for lr in learning_rates:
        # Nota: usiamo config.BATCH_SIZE e config.EPOCHS
        history = run_experiment(lr=lr, batch_size=config.BATCH_SIZE, epochs=config.EPOCHS)
        all_results[lr] = history

    # --- VISUALIZZAZIONE E SALVATAGGIO GRAFICO ---
    print("\n🎨 Generazione grafici comparativi...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for lr, data in all_results.items():
        ax1.plot(data['loss'], label=f'LR {lr}')
        ax2.plot(data['acc'], label=f'LR {lr}')

    ax1.set_title('Confronto Loss (Validation)')
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.set_title('Confronto Accuracy (Validation)')
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Accuracy %')
    ax2.legend()
    ax2.grid(True)

    # Salvataggio usando pathlib
    plot_path = config.OUTPUT_DIR / "comparison_results.png"
    plt.savefig(plot_path)
    print(f"✅ Grafico salvato in: {plot_path}")

    print("\n--- CLASSIFICA FINALE ---")
    for lr, data in all_results.items():
        print(f"LR: {lr} -> Miglior Accuracy: {max(data['acc']):.2f}%")