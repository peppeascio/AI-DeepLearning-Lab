"""
Analizzatore dei Pesi del Modello (Diagnostic Tool)
--------------------------------------------------
Questo script permette di ispezionare il contenuto di un file .pt 
estraendo i nomi dei layer e le dimensioni dei tensori associati.
"""

import torch
from pathlib import Path

# Importiamo il percorso del modello dal nostro config centrale
from config import MODEL_SAVE_PATH

def inspect_weights() -> None:
    """
    Carica lo stato del modello e stampa una tabella leggibile 
    della struttura dei parametri interni.
    """
    
    # 1. Controllo esistenza del file
    if not MODEL_SAVE_PATH.exists():
        print(f"❌ Errore: Il file dei pesi non esiste in {MODEL_SAVE_PATH}")
        print("💡 Suggerimento: Esegui prima l'addestramento con mnist_ai.py")
        return

    print(f"\n🔍 ISPEZIONE MODELLO: {MODEL_SAVE_PATH.name}")
    print("-" * 60)
    print(f"{'LAYER NAME':<30} | {'TENSOR SHAPE':<20}")
    print("-" * 60)

    try:
        # Carichiamo i pesi in CPU (più leggero per la sola ispezione)
        # weights_only=True è la pratica di sicurezza raccomandata da PyTorch
        state_dict = torch.load(MODEL_SAVE_PATH, map_location="cpu", weights_only=True)

        for layer_name, weights in state_dict.items():
            # Trasformiamo la forma del tensore in una stringa leggibile
            shape_str = str(list(weights.shape))
            print(f"{layer_name:<30} | {shape_str:<20}")

        print("-" * 60)
        print(f"✅ Ispezione completata. Totale parametri trovati: {len(state_dict)}")
        
    except Exception as e:
        print(f"❌ Errore durante la lettura del file: {e}")

if __name__ == "__main__":
    inspect_weights()