import torch
import torch.nn as nn
import torch.optim as optim
from src.config import DEVICE
from src.attacks import pgd_attack

def evaluate_model(model, loader, normalize_fn):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(normalize_fn(images))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def train_baseline(model, train_loader, val_loader, epochs, lr, normalize_fn, save_path):
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(normalize_fn(images))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        val_acc = evaluate_model(model, val_loader, normalize_fn)
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {100.*correct/total:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
    return model

def train_adversarial(model, train_loader, val_loader, epochs, lr, normalize_fn, save_path, eps, alpha, iters):
    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # PGD on the fly
            model.eval()
            adv_images, _, _ = pgd_attack(model, images, labels, eps, alpha, iters, normalize_fn)
            model.train()

            optimizer.zero_grad()
            outputs = model(normalize_fn(adv_images))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        val_acc = evaluate_model(model, val_loader, normalize_fn)
        print(f"Adv Epoch {epoch+1}/{epochs} | Adv Train Acc: {100.*correct/total:.2f}% | Clean Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
    return model
