import torch
import torch.nn as nn

def fgsm_attack(model, images, labels, epsilon, normalize_fn):
    model.eval()
    images_original = images.clone().detach().to(images.device)
    images_attack = images_original.clone().detach().requires_grad_(True)

    outputs = model(normalize_fn(images_attack))
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    if images_attack.grad is not None:
        images_attack.grad.zero_()
    loss.backward()

    grad_sign = images_attack.grad.detach().sign()
    images_adv = images_attack + epsilon * grad_sign
    images_adv = torch.clamp(images_adv, 0, 1).detach()

    return images_adv, images_adv - images_original, grad_sign

def pgd_attack(model, images, labels, epsilon, alpha, iters, normalize_fn):
    model.eval()
    images_original = images.clone().detach().to(images.device)
    images_adv = images_original.clone().detach()

    if epsilon > 0:
        images_adv = images_adv + torch.empty_like(images_adv).uniform_(-epsilon, epsilon)
        images_adv = torch.clamp(images_adv, 0, 1).detach()

    for _ in range(iters):
        images_adv.requires_grad_(True)
        outputs = model(normalize_fn(images_adv))
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        if images_adv.grad is not None:
            images_adv.grad.zero_()
        loss.backward()

        adv_images = images_adv + alpha * images_adv.grad.sign()
        eta = torch.clamp(adv_images - images_original, min=-epsilon, max=epsilon)
        images_adv = torch.clamp(images_original + eta, min=0, max=1).detach()

    return images_adv, images_adv - images_original, None
