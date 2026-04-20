import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Importiamo le nostre configurazioni e il modello
from config import *
from model import CifarNet

def plot_results(losses, accuracies):
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('Andamento Loss')
    plt.xlabel('Epoca')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Training Accuracy', color='green')
    plt.title('Andamento Accuratezza')
    plt.xlabel('Epoca')
    plt.legend()
    
    plt.savefig(OUTPUT_DIR / "training_metrics.png")
    print(f"📊 Grafici salvati in: {OUTPUT_DIR / 'training_metrics.png'}")

if __name__ == '__main__':
    print(f"🚀 Hardware in uso: {DEVICE}")

    # Pipeline di trasformazione (Usa le costanti di config.py)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    # Caricamento Dataset
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = CifarNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history_loss = []
    history_acc = []

    print(f"🔥 Inizio Training su CIFAR-10 ({EPOCHS} epoche)...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(trainloader, desc=f"Epoca [{epoch+1}/{EPOCHS}]")

        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=running_loss/len(trainloader), acc=f"{100.*correct/total:.2f}%")

        history_loss.append(running_loss/len(trainloader))
        history_acc.append(100.*correct/total)

    print(f"\n✅ Completato in {(time.time() - start_time)/60:.2f} minuti")
    
    # Salvataggi finali
    torch.save(model.state_dict(), MODEL_PATH)
    plot_results(history_loss, history_acc)