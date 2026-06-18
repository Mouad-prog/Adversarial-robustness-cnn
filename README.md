# Adversarial Robustness in Deep Learning (CNN)

Ce projet explore la vulnérabilité des réseaux de neurones convolutifs (CNN) face aux attaques adversariales (FGSM et PGD) et évalue l'efficacité de l'entraînement adversarial (Adversarial Training) comme mécanisme de défense. Le projet est évalué sur les datasets MNIST et CIFAR-10.

## 🏗️ Architecture du Code

Le code a été refactorisé depuis un Jupyter Notebook vers une architecture modulaire en Python :

- `src/config.py` : Configuration globale (chemins locaux, hyperparamètres, device, normalisation).
- `src/models.py` : Définition des architectures CNN (`MNISTNet` et `CIFAR10Net`).
- `src/data_loader.py` : Gestion des datasets et des DataLoaders.
- `src/attacks.py` : Implémentation des attaques FGSM (Fast Gradient Sign Method) et PGD (Projected Gradient Descent).
- `src/train.py` : Boucles d'entraînement classique et adversarial.
- `main.py` : Point d'entrée principal pour exécuter l'entraînement, les attaques et générer les évaluations.

## 🚀 Utilisation

1. Assurez-vous d'avoir installé les dépendances : `pip install torch torchvision matplotlib pandas`
2. Lancez le script principal : `python main.py`
