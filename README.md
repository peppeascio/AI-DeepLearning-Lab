# AI Deep Learning Lab 

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white" />
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  </a>
  <a href="https://developer.nvidia.com/cuda-zone">
    <img src="https://img.shields.io/badge/CUDA-RTX%203070-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  </a>
 
<p align="center"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](./LICENSE)
  </a>
</p>

- --

</p>

![Lab Banner](assets/img/Banner/banner_lab.png)

<p align="center">
  <a href="https://github.com/spiccillodev">
    <img src="https://img.shields.io/badge/SPICCILLO-CREATOR-black?style=for-the-badge&logo=github&logoColor=white" alt="Spiccillo Creator" />
  </a>
</p>

---

## Project Overview

Benvenuti nel mio laboratorio di ***Computer Vision***. Questo repository rappresenta un **ecosistema modulare** dedicato alla ricerca avanzata sull'efficienza delle <kbd>***CNN***</kbd> (_Convolutional Neural Networks_) ad alte prestazioni. Il progetto copre l'intero **ciclo di vita del modello**: dalla progettazione architetturale e analisi tensoriale, fino all'inferenza scalabile e all'implementazione di tecniche di **Hyperparameter Tuning** e un sistema di telemetria duale per il monitoraggio delle performance.

>[!TIP]
   > I benchmark dettagliati e i test sono disponibili nei Jupyter Notebook all'interno di ogni modulo.

  [![Jupyter Notebook](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](./01-MNIST-Digits/notebook/01_MNIST_Digits.ipynb)
  [![TensorBoard](https://img.shields.io/badge/Monitored%20with-TensorBoard-green?style=for-the-badge&logo=tensorflow&logoColor=white)](./01-MNIST-Digits/notebook/01_MNIST_Digits.ipynb)

> _Esempio:_  
> ![Model](https://img.shields.io/badge/Model-CNN%20Deep%20Learning-purple)
![General Overview](assets/img/General/convolutional_neural_network.png)

> _Riferimento progettuale:_ 
> <img src="https://img.shields.io/badge/Scope-MNIST%20%7C%20CIFAR10%20Experiments-orange">
![Esempio Analysis](assets/img/General/comprehensive_analysis.png)

---

##  Features 

<img src="https://img.shields.io/badge/Features-Experimental%20ML%20Lab-yellow">

   - **<u>GUI:</u>** Visualizzazione avanzata integrata con Matplotlib (Heatmap + Top 3 + Reasoning).
   - **<u>Logica di Ragionamento:</u>** Ogni classificazione include una spiegazione testuale dei pattern rilevati dal modello.
   - **<u>Smart Random Inference:</u>** Algoritmo di selezione file intelligente che evita la ripetizione delle immagini di test durante le demo.

---
## Specifiche Hardware & Performance Stack

<img src="https://img.shields.io/badge/GPU-Optimized%20Stack-green?logo=nvidia">
<img src="https://img.shields.io/badge/Performance-High%20Throughput-blue">
<img src="https://img.shields.io/badge/Architecture-RTX%2030%20%7C%2040%20Series-success">

>[!WARNING]
   > Sistema ottimizzato per hardware di fascia alta, capace di massimizzare il throughput dei tensori sfruttando la VRAM (fino a 10× rispetto alla CPU), riducendo i tempi di training e migliorando la reattività in inferenza. Testato con successo su GPU NVIDIA serie 30 e 40, garantendo versatilità e prestazioni stabili su più generazioni.

![Deep Learning GPU Stack](assets/img/Workstation/workstation.png)

| Componente | Specifiche Tecniche | Note sulle Prestazioni |
|---|---|---|
| _GPU Primaria_ | NVIDIA GeForce RTX 3070 (Ampere) | Elevata densità di CUDA Cores; utilizzata come dispositivo principale per il training |
| _GPU Secondaria_ | NVIDIA GeForce RTX 4060 (Ada Lovelace) | Supporto DLSS 3, architettura efficiente per inferenza |
| _Compute Capability_ |  8.6 (RTX 3070) / 8.9 (RTX 4060) | Compatibilità cross-generation con ottimizzazioni FP32 per stabilità numerica |
| _Deep Learning Stack_ | PyTorch + CUDA 12.1 + cuDNN | Accelerazione hardware nativa con pieno supporto alle GPU NVIDIA |
| _Ambiente Sviluppo_ | Anaconda / Python 3.11 | Gestione ambienti isolati e dipendenze ottimizzata |

---

## 📂 Repository Structure

Il progetto è organizzato in modo modulare per garantire scalabilità, separazione dei modelli e chiarezza dell’architettura.

```
AI-DeepLearning-Lab/
├── 01-MNIST-Digits/
├── 02-CIFAR10-Classifier/
└── assets/ 
```

###### Moduli principali
- [MNIST Digits](./01-MNIST-Digits)
  ![MNIST Module](https://img.shields.io/badge/Module_01-MNIST_Digits-blue?style=for-the-badge&logo=pytorch&logoColor=white)

- [CIFAR10 Classifier](./02-CIFAR10-Classifier) 
   ![Module_02-CIFAR10](https://img.shields.io/badge/Module_02-CIFAR10_Classifier-orange?style=for-the-badge&logo=googlephotos&logoColor=white)

- [assets](./assets) (Risorse grafiche per documentazione) 
  ![Assets](https://img.shields.io/badge/Assets-Visualization_%7C_Data-lightgrey?style=for-the-badge&logo=files&logoColor=white)

---
## Guida all'Installazione

Segui questi passaggi per configurare l'ambiente virtuale Python e installare tutte le dipendenze necessarie per eseguire i modelli e sfruttare l'accelerazione della GPU.

**1. Prerequisiti Hardware & Software**

- Driver NVIDIA aggiornati (per le serie RTX 30/40).
- [Anaconda](https://www.anaconda.com/) o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installato sul sistema.

**2. Creazione dell'Ambiente e Installazione Dipendenze**

Apri il terminale (o Anaconda Prompt), assicurati di essere posizionato all'interno della cartella principale del progetto (`AI-DeepLearning-Lab`) e avvia la configurazione:

```bash
# 1. Crea un ambiente virtuale isolato chiamato "ai_gpu" con Python 3.11
conda create --name ai_gpu python=3.11 -y

# 2. Attiva l'ambiente appena creato
conda activate ai_gpu

# 3. Installa automaticamente tutte le librerie necessarie
pip install -r requirements.txt
```

> [!TIP]
 > Verifica l'Accelerazione GPU:
Esegui questo test rapido per confermare che PyTorch "veda" la tua RTX:

```bash
python -c "import torch; print(f'GPU Disponibile: {torch.cuda.is_available()} - {torch.cuda.get_device_name(0)}')"
```

---

## Development Methodology

Il processo di creazione e validazione di ogni modello segue rigorosamente una pipeline standardizzata:

1. **Tensor Analysis:** Calibrazione delle dimensioni Input/outputs, normalizzazione e strategie di stochastic data augmentation.
2. **Model Architecture Design:** Progettazione di architetture CNN personalizzate, con attenzione alla profondità, al numero di filtri e alla scelta delle funzioni di attivazione (ReLU, Softmax).
3. **Loss Function & Optimization:** Implementazione di funzioni di perdita appropriate (Cross-Entropy) e ottimizzazione tramite algoritmi avanzati (Adam, SGD con Momentum).
4. **Data Loading & Preprocessing:** Utilizzo di DataLoader con batch size ottimizzati e tecniche di shuffling per garantire una distribuzione equilibrata dei dati durante il training.
5. **Validation & Testing:** Valutazione continua delle performance su set di validazione, con metriche chiave come Accuracy, Precision, Recall e F1-Score per monitorare la generalizzazione del modello.
6. **Hyperparameter Tuning:** Sperimentazione sistematica di learning rates, batch sizes e architetture per identificare la configurazione ottimale.
7. **Iterative Training:** Monitoraggio della convergenza tramite **TensorBoard**, utilizzando il tracker di progresso [<kbd>tqdm</kbd>](https://github.com/tqdm/tqdm) e l'ottimizzatore *Adam*.

8. **Brain Scanning:** Estrazione e visualizzazione delle mappe di attivazione per verificare  cosa i filtri convoluzionali stiano "percependo" a diverse profondità *(feature extraction)*.
9.  **Persistent Logging:** Registrazione storica di ogni predizione, punteggio di confidenza ed errore in formato CSV per analisi post-training.
10. **Model Export:** Conversione in formato [![ONNX](https://img.shields.io/badge/ONNX-Neural%20Network%20Exchange-blue?style=flat-square&logo=onnx&logoColor=white)](https://onnx.ai/) per la visualizzazione universale del grafo e il deployment.

><mark>*Esempio di monitoraggio della convergenza (Loss e Accuracy) via TensorBoard.*</mark>
![Tensorboard curves](assets/img/General/Tensorboard_curves.png)

---
## Modelli e Ricerca Applicata

#### 🔢 1. MNIST: Handwritten Digit Recognition
##### Structure Project

>📂 01-MNIST-Digits/
├── 📁 [data/](./01-MNIST-Digits/data) - Dataset MNIST (auto-download)
│   └── [.gitkeep](./01-MNIST-Digits/data/.gitkeep)
├── 📁 [models/](./01-MNIST-Digits/models) - Pesi del modello salvati
│   ├── [.gitkeep](./01-MNIST-Digits/models/.gitkeep)
│   └── [mnist_classifier_v1.pt](./01-MNIST-Digits/models/mnist_classifier_v1.pt)
├── 📁 [netron_visualization/](./01-MNIST-Digits/netron_visualization) - Architettura universale
│   ├── 🌐 [mnist_model_graph.onnx](./01-MNIST-Digits/netron_visualization/mnist_model_graph.onnx)
│   └── 📄 [mnist_model_graph.onnx.data](./01-MNIST-Digits/netron_visualization/mnist_model_graph.onnx.data)
├── 📁 [notebook/](./01-MNIST-Digits/notebook) - Sperimentazione interattiva
│   └── [01_MNIST_Digits.ipynb](./01-MNIST-Digits/notebook/01_MNIST_Digits.ipynb)
├── 📁 [outputss/](./01-MNIST-Digits/outputss) - Risultati del training e grafici
│   ├── 📁 [LR_0.01_BS_256/](./01-MNIST-Digits/outputss/LR_0.01_BS_256) - Log iperparametri
│   ├── 📁 [LR_0.001_BS_256/](./01-MNIST-Digits/outputss/LR_0.001_BS_256) - Log iperparametri
│   ├── 📁 [LR_0.0001_BS_256/](./01-MNIST-Digits/outputss/LR_0.0001_BS_256) - Log iperparametri
│   └── [comparison_results.png](./01-MNIST-Digits/outputss/comparison_results.png)
├── 📁 [src/](./01-MNIST-Digits/src) - Codice sorgente principale
│   ├── ⚙️ [config.py](./01-MNIST-Digits/src/config.py) - Impostazioni centralizzate
│   ├── [export_to_netron.py](./01-MNIST-Digits/src/export_to_netron.py) - Converter ONNX
│   ├── [inspect_model.py](./01-MNIST-Digits/src/inspect_model.py) - Analisi dei tensori
│   ├── [mnist_ai.py](./01-MNIST-Digits/src/mnist_ai.py) - Core training script
│   └── [predict_mnist_heatmap.py](./01-MNIST-Digits/src/predict_mnist_heatmap.py) - Inferenza visiva
├── 📁 [test/](./01-MNIST-Digits/test) - Immagini per test manuali
│   └── [5.png](./01-MNIST-Digits/test/5.png)
└── 📄 [requirements.txt](./01-MNIST-Digits/requirements.txt) - Dipendenze del modulo


- **Dataset:** 60.000 immagini di training e 10.000 di test, dimensioni 28x28 pixel in scala di grigi.
- **Architecture:** `Dual Conv2d` + `Dual MaxPool` + `Linear Classification Head`.

- **Technical Insight:** Il modello gestisce una transizione da una spatial feature map 2D a un flattened vector da **7.744 dimensioni**, ridotto poi a **128 neuroni latenti** per la decisione finale.

- **Visual Audit:** Studio approfondito dei pesi dello strato `conv.0.weight` tramite Netron.
<br><br>
#### Caratteristiche Principali:

***Auto-Cleanup***: La cartella [outputss](./01-MNIST-Digits/outputss) viene ottimizzata automaticamente a ogni avvio per evitare accumuli di spazio.

***Hyperparameter Testing***: Il sistema testa sequenzialmente diversi Learning Rates (es. 0.01, 0.001, 0.0001) per trovare il setup ideale.

***Dual Visualization***: Matplotlib: Generazione automatica di un grafico comparativo [comparison_results.png](./01-MNIST-Digits/outputss/comparison_results.png) per una consultazione rapida senza dipendenze esterne.

><mark>
> Sotto un esempio di inferenza a bassa latenza con generazione dinamica di heatmap.
 </mark>

![MNIST Demo](assets/img/MNIST/demo3.png)
![Inference Demo](assets/img/MNIST/3.gif)
><mark>
> Inference Demo:
> Riconoscimento cifre e analisi dei gradienti in tempo reale.
 </mark>

---

### 2. CIFAR-10: Elite Object Recognition
**Structure Project**
> 📂 [**02-CIFAR10-Classifier/**]
├── 📁 [data/](./02-CIFAR10-Classifier/data) - Dataset CIFAR-10 (Ignorato da Git)
│   └── [.gitkeep](./02-CIFAR10-Classifier/data/.gitkeep)
├── 📁 [models/](./02-CIFAR10-Classifier/models) - Modelli PyTorch salvati
│   ├── [.gitkeep](./02-CIFAR10-Classifier/models/.gitkeep)
│   └── [modello_cifar10.pt](./02-CIFAR10-Classifier/models/modello_cifar10.pt)
├── 📁 [notebook/](./02-CIFAR10-Classifier/notebook) - Analisi e prototipazione
│   └── 📓 [CIFAR-10.ipynb](./02-CIFAR10-Classifier/notebook/CIFAR-10.ipynb)
├── 📁 [outputss/](./02-CIFAR10-Classifier/outputss) - Report e metriche
│   ├── 📁 [logs/](./02-CIFAR10-Classifier/outputss/logs) - Storico predizioni in CSV/TXT
│   ├── 📁 [predictions/](./02-CIFAR10-Classifier/outputss/predictions) - Dashboard e grafici XAI
│   ├── [training_metrics.png](./02-CIFAR10-Classifier/outputss/training_metrics.png) - Curve di Loss/Accuracy
│   └── [.gitkeep](./02-CIFAR10-Classifier/outputss/.gitkeep)
├── 📁 [src/](./02-CIFAR10-Classifier/src) - Logica di addestramento e inferenza
│   ├── ⚙️ [config.py](./02-CIFAR10-Classifier/src/config.py) - Configurazione iperparametri
│   ├── [model.py](./02-CIFAR10-Classifier/src/model.py) - Architettura della CNN
│   ├── [train.py](./02-CIFAR10-Classifier/src/train.py) - Training loop primario
│   ├── [predict_plot.py](./02-CIFAR10-Classifier/src/predict_plot.py) - Inferenza visiva con Matplotlib
│   └── [predict_terminal.py](./02-CIFAR10-Classifier/src/predict_terminal.py) - Inferenza leggera via terminale
├── 📁 [test/](./02-CIFAR10-Classifier/test) - Immagini reali per validazione manuale
│   └── 📁 [img/](./02-CIFAR10-Classifier/test/img) - Campioni (aereo, auto, gatto, ecc.)
└── 📄 [requirements.txt](./02-CIFAR10-Classifier/requirements.txt) - Dipendenze specifiche modulo

Sfida di riconoscimento su 10 classi (RGB 32x32), potenziata da tecniche di **Interpretability**.

***Dataset***: 50.000 immagini fotografiche di training e 10.000 di test divise in 10 categorie (aerei, auto, uccelli, gatti, ecc.). A differenza di MNIST, i pattern geometrici qui sono immersi in sfondi rumorosi e texture complesse.

***Architecture***: CNN Avanzata con moduli di regolarizzazione (Dropout), normalizzazione spaziale (BatchNorm2d) per stabilizzare i gradienti e canali di convoluzione profondi progettati per l'estrazione di feature tridimensionali (RGB).

***Technical Insight***: Gestione nativa di tensori a 3 canali di colore. Il modello impara a discriminare forme complesse costruendo gerarchie di caratteristiche: dai bordi semplici nei primi layer, fino alla composizione di musi di animali o carrozzerie negli strati più profondi.
<br><br>
#### Caratteristiche Principali:

- **XAI (Explainable AI):** Implementazione di **Grad-CAM** per generare heatmap dinamiche che evidenziano le aree decisionali.

- **Ambiguity Detection:** Algoritmo per il calcolo del *Confidence Gap* tra classi visivamente simili (es. Auto vs Camion).
- **Persistent CSV Logging:** Tracciamento automatico di ogni singola immagine testata in un file di log (classificazioni_log.csv) per monitorare l'accuratezza in fase di produzione.
<br><br>

![Grad-CAM_demo](assets/img/CIFAR10/Grad-CAM_demo.png)

> [!TIP]
   >*Analisi Grad-CAM: Il modello distingue tra Auto e Camion analizzando feature specifiche.*

---

## Come Eseguire i Modelli (Quick Start)
Se vuoi saltare direttamente all'azione e testare le reti neurali già addestrate con le immagini presenti nella cartella test/, puoi usare gli script pronti all'uso:

> **Per testare MNIST (Cifre scritte a mano):**

```bash
cd 01-MNIST-Digits
#### Per l'interfaccia grafica con ragionamento
python src/predict_mnist_heatmap.py
# Mostra architettura del modello e analisi dei tensori
python src/inspect_model.py
```

   > **Per testare CIFAR-10 (Oggetti e Animali):**

```bash
cd 02-CIFAR10-Classifier
# Per la GUI che mostra l'immagine, la top-3 delle classi e le heatmap
python src/predict_plot.py
# Per l'outputs su terminale con salvataggio log CSV
python src/predict_terminal.py
```

---

### Performance & Visual Benchmarks
#### [01-MNIST-Digits.ipynb](./01-MNIST-Digits/notebook/01_MNIST_Digits.ipynb)
<p align="center">
<img src="assets/img/benchmark/optimizer_comp.png" width="80%" alt="Optimizer Comparison">


<sub><b>Confronto Ottimizzatori:</b> Analisi della velocità di convergenza tra Adam, SGD e RMSprop.</sub>
</p>

<p align="center">
<img src="assets/img/benchmark/lr_impact.png" width="80%" alt="Learning Rate Impact">


<sub><b>Impact Analysis:</b> Studio della stabilità del gradiente al variare del Learning Rate.</sub>
</p>

<p align="center">
<img src="assets/img/benchmark/xai.png" width="80%" alt="Explainable AI">


<sub><b>XAI & Activation Mapping:</b> Visualizzazione dei pattern rilevati dai filtri convoluzionali (Brain Scanning).</sub>
</p>

<p align="center">
<img src="assets/img/benchmark/inferenza_custom.png" width="80%" alt="Custom Inference">


<sub><b>Real-world Test:</b> Validazione del modello su cifre scritte a mano fuori dal dataset originale.</sub>
</p>

---

## ***Feature in Sviluppo***

- [ ] **ResNet-50 Integration:** Implementazione di Skip-Connections per superare il limite delle 50 epoche di training senza degradazione.

- [ ] **Advanced Data Augmentation:** Introduzione di Color Jittering e Random Flipping per aumentare la robustezza dei modelli.

- [ ] **Real-time Camera Lab:** Overlay delle heatmap Grad-CAM in tempo reale direttamente via webcam.

<p align="center">
  <a href="https://github.com/spiccillodev">
    <img src="https://img.shields.io/badge/%20Sviluppato%20con%20dedizione-Spiccillo-blueviolet?style=for-the-badge&logo=github&logoColor=white" alt="Spiccillo Profile" />
  </a>
</p>