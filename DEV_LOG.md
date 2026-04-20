# Engineering Dev-Log | AI Deep Learning Lab

Questo documento costituisce il registro tecnico dell'evoluzione architetturale del laboratorio. Non è una semplice lista di modifiche, ma una documentazione delle **scelte ingegneristiche** effettuate per garantire che ogni modello di Deep Learning sia scalabile, portabile e pronto per una produzione ipotetica.

- --

## Core Engineering Standards (Consolidati nel Modulo 01)

*Standard obbligatori applicati per mitigare il debito tecnico e garantire la manutenibilità.*

### 1\. Architettura Disaccoppiata (Decoupled Design)

Il sistema è diviso in tre domini logici separati per minimizzare le dipendenze incrociate:

-    **Domain 01: Configuration (`config.py`)**  
    Unica *Single Source of Truth (SSoT)* per:
    -    iperparametri
    -    percorsi
    -    costanti di normalizzazione
-    **Domain 02: Model Definition**  
    La logica dei layer e dei pesi è isolata in classi dedicate (es. `DigitNet`, `CifarNet`) per essere riutilizzata senza side-effects.
-    **Domain 03: Execution**  
    Script di training e inferenza che consumano i moduli senza ridefinire la logica.

- --

### 2\. Gestione dei Percorsi con Agnosticismo OS

Sostituzione completa delle stringhe hardcoded con `pathlib`.

-    **Logica**
    
    BASE\_DIR \= Path(\_\_file\_\_).resolve().parent.parent
    
-    **Vantaggio**
    -    Compatibilità immediata tra:
        -    Windows (RTX 3070 Desktop)
        -    Linux (Server / Cloud)
        -    macOS
-    **Validazione reale**
    -    Migrazione testata da Google Drive → filesystem locale
    -    Nessuna modifica manuale richiesta

- --

### 3\. Rigore Sintattico & Type Safety

-    **Type Hinting**
    
    def predict(img: Image.Image) -> int:
    
-    **Gestione PyTorch dinamico**
    -    Type hints espliciti (`torch.Tensor`)
    -    Uso mirato di `# type: ignore`
-    **Obiettivo**
    -    Zero warning Pylance
    -    Workspace pulito e mantenibile

- --

## Sfide Risolte: Migrazione Hardware & Stabilità

### 1\. OpenMP Error #15

-    **Problema**  
    Conflitto tra Intel MKL (NumPy/SciPy) e runtime OpenMP PyTorch
-    **Scelta ingegneristica**  
    ❌ Evitato fix nel codice (`os.environ`)  
    ✅ Fix a livello ambiente
-    **Soluzione**
    
    conda env config vars set KMP\_DUPLICATE\_LIB\_OK\=TRUE

- **OpenMP Stability Fix (KMP_DUPLICATE_LIB_OK)**: configurata variabile di ambiente a livello di sistema (`KMP_DUPLICATE_LIB_OK=TRUE`) per risolvere conflitti di runtime tra librerie OpenMP (Intel MKL vs PyTorch), evitando duplicazioni della libreria `libiomp5md.dll`.
    

- --

### 2\. Dependency Resolution (Protobuf & ONNX)

-    **Problema**  
    Conflitto tra `protobuf` e `onnx`
-    **Soluzione**
    -    Upgrade a `protobuf >= 4.25`
    -    Rimozione vincoli rigidi

- --

### 3\. Encoding & Unicode Stability

-    **Problema**  
    `cp1252` non supporta emoji/log Unicode
-    **Soluzione**
    
    open(file, encoding\="utf-8")
    

- --

## 📂 Modulo 01: MNIST Digits - Refactoring & Optimization

### Problemi iniziali (Legacy)

-    Path assoluti (`Z:\`, `G:\`)
-    Model drift (duplicazione classi)
-    Preprocessing incoerente

- --

### Soluzioni Implementate

| Area | File / Script | Descrizione |
| --- | --- | --- |
| Configurazione | src/config.py | Controllo centralizzato |
| Inferenza | Import DigitNet | Eliminazione mismatch |
| Diagnostica | inspect_model.py | Analisi tensori |
| Visualizzazione | export_to_netron.py | Grafi ONNX |

- --

### Normalizzazione

$$
x' = \frac{x - 0.1307}{0.3081}
$$

- --

## 🟦 Modulo 02: Deep Learning Image Classification (CIFAR-10)

**Status:**   Production Ready  
**Hardware Target:** NVIDIA RTX 3070 (Ampere) / 4060 (Ada Lovelace)

- --

### 1\. Software Architecture

#### 1.1 Decoupling & Asset Management

-    **`/src`**
    -    Codice puro
    -    `config.py` = SSoT
-    **Separazione asset**
    -    `/data` → dataset
    -    `/models` → checkpoint
-    **Vantaggi**
    -    Repo Git leggero
    -    Pipeline production-ready

- --

#### 1.2 Inference State & Persistence

Sistema di memoria a breve termine:

-    File: `outputs/logs/prediction_history.txt`
-    Tiene traccia degli ultimi **3 input**

**Algoritmo**

-    Filtro di esclusione
-    Sampling pseudo-casuale
-    Zero ripetizioni visive

- --

### 2\. Model Engineering: CifarNet

#### 2.1 Preprocessing

$$
x_{norm} = \frac{x - \mu}{\sigma}
$$

- $\mu_{RGB} = [0.4914, 0.4822, 0.4465]$
- $\sigma_{RGB} = [0.2023, 0.1994, 0.2010]$

- --

#### 2.2 Architettura CNN

-    **Feature Extraction**
    -    3 blocchi conv
    -    kernel 3×3
    -    padding 1
-    **Downsampling**
    -    `MaxPool2d(2,2)`
-    **Regularization**
    -    `Dropout(0.25)`

- --

## Technical Problem Solving (Fix Log)

| Issue | Root Cause | Soluzione |
| --- | --- | --- |
| OpenMP Error | Conflitto MKL vs PyTorch | KMP_DUPLICATE_LIB_OK=TRUE |
| Encoding Crash | cp1252 incompatibile | UTF-8 |
| Type Issues | Limiti Pylance | Type hints + ignore |

- --

## 🔍 XAI: Explainable AI

### Grad-CAM

-    **Hook**
    -    Forward + Backward
    -    Layer: `conv3`
-    **Peso canali**

$$
\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial Y^c}{\partial A_{ij}^k}
$$

-    **Output**
    -    Heatmap + ReLU
    -    Overlay JET (OpenCV)
    -    ROI evidenziate

- --

### Dashboard (`predict_plot.py`)

-    Heatmap Grad-CAM
-    Top-3 probabilità
-    Confronto dataset reale

- --

## 📊 Operations & Benchmarking

-    **Logging**
    -    CSV
    -    timestamp + confidence + label
-    **GPU Optimization**
    -    `.to(DEVICE)`
    -    riduzione bottleneck CPU→GPU

- --

## 🚀 Roadmap

-    ResNet-50 (skip connections)
-    Data Augmentation avanzata
-    Grad-CAM real-time (webcam)

- --

## 📅 Status

-    **Ultimo aggiornamento:** 20 Aprile 2026
-    **Hardware:** RTX 3070 / 4060
-    **Stato:** Modulo 01 & 02 stabilizzati