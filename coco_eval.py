import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import wandb

from dataloader import get_coco_segmentation_dataloaders

def evaluate_segmentation(model, data_loader, num_classes, device, use_amp = False, ignore_index = 255):
    if isinstance(model, str):
        model = torch.load(model, map_location=device)
    model = model.to(device)
    model.eval()

    confmat = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    if use_amp and torch.cuda.is_available():
        autocast = torch.autocast('cuda', dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating segmentation', dynamic_ncols=True):
            if isinstance(batch, (list, tuple)):
                images, masks = batch
            elif isinstance(batch, dict):
                images, masks = batch['images'], batch['masks']
            else:
                images, masks = batch[0]['images'], batch[0]['masks']

            images = images.to(device, non_blocking=True)
            if images.ndim == 4 and images.shape[1] not in (1, 3) and images.shape[-1] in (1, 3):
                images = images.permute(0, 3, 1, 2)

            masks = masks.long().to(device, non_blocking=True)

            with autocast:
                logits = model(images)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[-2:], mode='bilinear', align_corners=False
                )

            preds = logits.argmax(1)

            masks_flat = masks.view(-1)
            preds_flat = preds.view(-1)

            if ignore_index is not None:
                valid = masks_flat != ignore_index
                masks_flat = masks_flat[valid]
                preds_flat = preds_flat[valid]

            k = (masks_flat >= 0) & (masks_flat < num_classes)
            labels = num_classes * masks_flat[k] + preds_flat[k]
            bincount = torch.bincount(labels, minlength=num_classes ** 2)
            confmat += bincount.view(num_classes, num_classes)

    tp = confmat.diag()
    total = confmat.sum()

    if total > 0:
        acc_global = (tp.sum().float() / total.float()).item()
    else:
        acc_global = 0.0

    acc_cls = tp.float() / confmat.sum(1).clamp(min=1).float()
    acc_cls_mean = acc_cls.mean().item()

    iu = tp.float() / (confmat.sum(1) + confmat.sum(0) - tp).clamp(min=1).float()
    miou = iu.mean().item()

    metrics = {'mIoU': miou, 'pixel_acc': acc_global, 'mean_class_acc': acc_cls_mean, 'per_class_iou': iu.detach().cpu().tolist()}
    return metrics

def coco_segmentation_finetune(model, coco_root, num_classes, batch_size = 8, size = 224, 
                               num_workers = 4, max_epochs = 20, lr = 1e-3, weight_decay = 1e-4,
                               ignore_index = 255, use_amp = False):
    train_loader, val_loader = get_segmentation_dataloaders(root = coco_root, batch_size = batch_size, size = size,
                                                                     num_workers = num_workers)

    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    if use_amp and torch.cuda.is_available():
        autocast = torch.autocast('cuda', dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()

    best_miou = 0.0
    history = {'train_loss': [], 'val_mIoU': [], 'val_pixel_acc': []}

    wandb.init(project='theseus_coco_seg', name=args.exp_name, config={
        'lr': lr,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'num_classes': num_classes,
    })

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        for step, (images, masks) in enumerate(
            tqdm(train_loader, desc=f'COCO seg: epoch {epoch}', dynamic_ncols=True)
        ):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).long()

            optimizer.zero_grad(set_to_none=True)

            with autocast:
                logits = model(images)  # [B, C, H', W']
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = torch.nn.functional.interpolate(
                        logits,
                        size=masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(1, len(train_loader))
        history['train_loss'].append(avg_train_loss)

        val_metrics = evaluate_segmentation(model, val_loader, num_classes = num_classes, device = device, use_amp = use_amp, 
                                            ignore_index = ignore_index,)

        history['val_mIoU'].append(val_metrics['mIoU'])
        history['val_pixel_acc'].append(val_metrics['pixel_acc'])
        best_miou = max(best_miou, val_metrics['mIoU'])

        wandb.log({'epoch': epoch, 'seg_train_loss': avg_train_loss, 'seg_val_mIoU': val_metrics['mIoU'], 
                    'seg_val_pixel_acc': val_metrics['pixel_acc'],})

        print(
            f'[Epoch {epoch}] train_loss={avg_train_loss:.4f}, '
            f'val_mIoU={val_metrics['mIoU']:.4f}, '
            f'val_pixel_acc={val_metrics['pixel_acc']:.4f}'
        )

    wandb.finish()

    return {
        'history': history,
        'best_mIoU': best_miou,
        'final_metrics': val_metrics,
    }

def coco_segmentation_eval(model, coco_root, num_classes, batch_size = 8, size = 224, 
                           num_workers = 4, device = device, use_amp = False, 
                           ignore_index = 255):
                        
    _, val_loader = get_coco_segmentation_dataloaders(root = coco_root, batch_size = batch_size, size = size, num_workers = num_workers)
    return evaluate_segmentation(model, val_loader, num_classes=num_classes, device=device, use_amp=use_amp, ignore_index=ignore_index)