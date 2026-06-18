import torch
import torch.nn as nn

def fgsm_attack(model, images, labels, epsilon, normalize_fn):
    model.eval()
    images_original = images.clone().detach()
    images_attack = images_original.clone().detach()
    images_attack.requires_grad_(True)

    outputs = model(normalize_fn(images_attack))
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    if images_attack.grad is not None:
        images_attack.grad.zero_()
    loss.backward()

    grad_sign = images_attack.grad.detach().sign()
    images_adv = images_attack + epsilon * grad_sign
    images_adv = torch.clamp(images_adv, 0, 1).detach()
    perturbation = images_adv - images_original

    return images_adv, perturbation, grad_sign
