import torch
import torch.nn as nn
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from src.utils import set_seeds
from src.focal_loss import FocalLoss, reweight
from torchvision.ops import sigmoid_focal_loss
import numpy as np
import wandb
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

def setup_training(
        model, criterion='cross_entropy', optimizer="sgd", 
        lr=0.001, momentum=0.9, gamma=2.0, alpha=1.0, weight_decay=0.01,
        device='cpu', cls_num_list=None):
    """
    Set up the optimizer and loss function based on the training configuration.
    Args:
        model: The model to train.
        criterion: Loss function ('cross_entropy' or 'focal').
        optimizer: Optimizer type ('sgd').
        lr: Learning rate.
        momentum: Momentum for SGD.
        gamma: Focusing parameter for focal loss.
        alpha: Weighting factor for focal loss.
    Returns:
        crit: Loss function.
        optim: Optimizer object.
    """
    if criterion == 'cross_entropy':
        crit = CrossEntropyLoss()
    elif criterion == 'focal':
        if alpha == 'reweight':
            # print("hi")
            alpha = reweight(cls_num_list, beta=0.9999)
            # print(alpha)
        crit = FocalLoss(gamma=gamma, weight=alpha, device=device)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    if optimizer == 'sgd':
        optim = SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
        )
    elif optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters(),weight_decay=weight_decay,lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    return crit, optim

#################

def train(model, train_loader, criterion, optimizer, epoch, config, device='cpu'):
    """
    Train the model for one epoch.
    Returns:
        float: Average loss for all batches in the epoch. Summarizing how 
        the model is performing during this epoch
    """
    print(f"Starting training for epoch {epoch+1}")
    model.train()
    total_loss = 0
    # total_steps = len(train_loader) # num batches in loader
    # tracking_loss = []  # List to store loss for every batch

    for batch_n, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        # tracking_loss.append(loss.item())  # Store batch loss for tracking 

        # backward and optimize
        optimizer.zero_grad()
        loss.backward() # wrt
        optimizer.step()

        # ✨ W&B: Log loss over training steps, visualized in the UI live
        if (batch_n + 1 ) % 50 == 0: 
            wandb.log({"loss" : loss, "epoch": epoch+1,})


        # Print progress every 100 batches
        if (batch_n + 1 ) % 100 == 0: 
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.12f}'
                .format(epoch+1, config["train"]["epochs"], batch_n+1, len(train_loader), loss.item()))
            
    avg_loss = total_loss / len(train_loader)
    return avg_loss #,tracking_loss

###########

def log_preds(
        images, image_ids, labels, probs, predicted, test_table):
    """
    Log predictions, confidence scores, and true labels for a batch of test 
    images. Creating a table of results in W&B. It visualizes image 
    misclassifications.
    """
    
    log_scores = probs.cpu().numpy()  # Convert to NumPy for logging
    log_images = images.cpu().numpy()
    log_labels = labels.argmax(axis=1).cpu().numpy()
    log_preds = predicted.cpu().numpy()

    for img, img_id, true_label, pred, conf_scores in zip(
        log_images, image_ids, log_labels, log_preds, log_scores
    ):
            
        # Convert image from [C, H, W] to [H, W, C]
        img = img.transpose(1, 2, 0)
        
        test_table.add_data(
            img_id,              # Image_ID
            wandb.Image(img),    # Image
            pred + 1,            # Predicted class (indexing starts at 0)
            true_label + 1,      # Ground-truth label
            *conf_scores         # Confidence scores for all classes
        )
        break


def evaluate(
        model, val_loader, criterion, config, epoch, log_counter=0, device='cpu'):
    """
    Evaluate the model on a test/validation dataset.
    Returns: dict: Metrics including accuracy and average loss.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    # Collect predictions and labels
    all_preds, all_labels = [], []

    # ✨ Initialize the W&B table
    test_table = wandb.Table(columns=[
        "image_id", "image", "predicted", "true_label",
        "1antelope_duiker", "2bird", "3blank",
        "4civet_genet", "5hog", "6leopard",
        "7monkey_prosimian", "8rodent"
    ])
    
    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            image_ids = batch["image_id"]

            # forward pass
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)  # For logging

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1) 
            correct += (predicted == labels.argmax(dim=1)).sum().item()
            total += labels.size(0)

            # Store predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.argmax(dim=1).cpu().numpy())  # Convert one-hot to class indices
            
            # Log predictions (only log a limited number of batches)
            if epoch == config["train"]["epochs"] and log_counter <= config["log"]["img_count"]:
                log_preds(images, image_ids, labels, 
                                     probs, predicted,test_table)
                log_counter += 1  

    # Calculate metrics
    acc = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    # conf_matrix = confusion_matrix(all_labels, all_preds) #not using

    class_report = classification_report(
        all_labels, 
        all_preds, 
        target_names=[
            "antelope_duiker", "bird", "blank", "civet_genet", 
            "hog", "leopard", "monkey_prosimian", "rodent"
        ],
        output_dict=True,
        zero_division=0
    )

    # ✨ Log class-wise F1 scores to W&B
    for class_name, metrics in class_report.items():
        if isinstance(metrics, dict):  # Ignore overall metrics like 'accuracy'
            wandb.log({
                f"f1_{class_name}": metrics["f1-score"]
            })

    # ✨ Log metrics to W&B
    wandb.log({
        "eval_loss": avg_loss,
        "eval_accuracy": acc,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_macro_f1": macro_f1,
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=[
                "antelope_duiker", "bird", "blank", "civet_genet", 
                "hog", "leopard", "monkey_prosimian", "rodent"
            ]
        )
    })
    wandb.log({"test_predictions": test_table})
        
    print(f"Eval - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, MacroF1: {macro_f1:.2f}")
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "all_preds": all_preds,
        "all_labels": all_labels
    }
  

def evaluate_low_log(
        model, val_loader, criterion, config, epoch, log_counter=0, device='cpu'):
    """
    Evaluate the model on a test/validation dataset.
    Returns: dict: Metrics including accuracy and average loss.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    # Collect predictions and labels
    all_preds, all_labels = [], []

    # Initialize confusion matrix
    num_classes = config["model"]["num_classes"]
    class_names = [
        "antelope_duiker", "bird", "blank", "civet_genet", 
        "hog", "leopard", "monkey_prosimian", "rodent"
    ]
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1) 
            correct += (predicted == labels.argmax(dim=1)).sum().item()
            total += labels.size(0)
            # Store predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.argmax(dim=1).cpu().numpy())  # Convert one-hot to class indices
            
            
    # Calculate metrics
    acc = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    # conf_matrix = confusion_matrix(all_labels, all_preds) #not using

    class_report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # ✨ Log class-wise F1 scores to W&B
    for class_name, metrics in class_report.items():
        if isinstance(metrics, dict):  # Ignore overall metrics like 'accuracy'
            wandb.log({
                f"f1_{class_name}": metrics["f1-score"]
            })

    # ✨ Log metrics to W&B
    wandb.log({
        "eval_loss": avg_loss,
        "eval_accuracy": acc,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_macro_f1": macro_f1,
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=class_names
        )
    })
        
    print(f"Eval - Loss: {avg_loss:.4f}, "
          f"Accuracy: {acc:.2f}%, Precision: {precision:.2f}, "
          f"Recall: {recall:.2f}, F1: {f1:.2f}, MacroF1: {macro_f1:.2f}")
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "all_preds": all_preds,
        "all_labels": all_labels
    }

def train_deit(
        model, train_loader, criterion, optimizer, 
        epoch, config, device='cpu'):
    """
    Train the model for one epoch.
    Returns:
        float: Average loss for all batches in the epoch. Summarizing how 
        the model is performing during this epoch
    """
    print(f"Starting training for epoch {epoch+1}")
    model.train()
    total_loss = 0

    for batch_n, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
 

        # forward pass
        outputs = model(pixel_values=images)
        loss = criterion(outputs.logits, labels)
        total_loss += loss.item() 

        # backward and optimize
        optimizer.zero_grad()
        loss.backward() # wrt
        optimizer.step()

        # ✨ W&B: Log loss over training steps, visualized in the UI live
        if (batch_n + 1 ) % 50 == 0: 
            wandb.log({"loss" : loss, "epoch": epoch+1,})


        # Print progress every 100 batches
        if (batch_n + 1 ) % 100 == 0: 
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.12f}'
                .format(epoch+1, config["train"]["epochs"], batch_n+1, len(train_loader), loss.item()))
            
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate_deit(
        model, val_loader, criterion, config, epoch, device='cpu'):
    """
    Evaluate the model on a test/validation dataset.
    Returns: dict: Metrics including accuracy and average loss.
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    # Collect predictions and labels
    all_preds, all_labels = [], []

    # Initialize confusion matrix
    num_classes = config["model"]["num_classes"]
    class_names = [
        "antelope_duiker", "bird", "blank", "civet_genet", 
        "hog", "leopard", "monkey_prosimian", "rodent"
    ]
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # forward pass
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.logits, dim=1)
            correct += (predicted == labels.argmax(dim=1)).sum().item()
            total += labels.size(0)
            # Store predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.argmax(dim=1).cpu().numpy())  # Convert one-hot to class indices
            
            
    # Calculate metrics
    acc = 100 * correct / total
    avg_loss = total_loss / len(val_loader)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    # conf_matrix = confusion_matrix(all_labels, all_preds) #not using

    class_report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # ✨ Log class-wise F1 scores to W&B
    for class_name, metrics in class_report.items():
        if isinstance(metrics, dict):  # Ignore overall metrics like 'accuracy'
            wandb.log({
                f"f1_{class_name}": metrics["f1-score"]
            })

    # ✨ Log metrics to W&B
    wandb.log({
        "eval_loss": avg_loss,
        "eval_accuracy": acc,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_macro_f1": macro_f1,
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=class_names
        )
    })
        
    print(f"Eval - Loss: {avg_loss:.4f}, "
          f"Accuracy: {acc:.2f}%, Precision: {precision:.2f}, "
          f"Recall: {recall:.2f}, F1: {f1:.2f}, MacroF1: {macro_f1:.2f}")
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "all_preds": all_preds,
        "all_labels": all_labels
    }
############## currently not in use



def setup_training_two_stage(
        model, criterion='cross_entropy', optimizer="sgd", 
        lr_head=0.001, lr_backbone=0.0001, momentum=0.9, gamma=2.0, alpha=1.0, weight_decay=0.01,
        device='cpu', cls_num_list=None, stage=1):
    """
    Set up the optimizer and loss function based on the training configuration.
    Args:
        model: The model to train.
        criterion: Loss function ('cross_entropy' or 'focal').
        optimizer: Optimizer type ('sgd', 'adam').
        lr_head: Learning rate for the classification head.
        lr_backbone: Learning rate for the backbone.
        momentum: Momentum for SGD.
        gamma: Focusing parameter for focal loss.
        alpha: Weighting factor for focal loss.
        weight_decay: Regularization term.
        device: Device to use ('cpu', 'cuda').
        cls_num_list: Class count for reweighting.
        stage: Training stage (1 for head-only, 2 for fine-tuning).
    Returns:
        crit: Loss function.
        optim: Optimizer object.
    """

    if criterion == 'cross_entropy':
        crit = CrossEntropyLoss()
    elif criterion == 'focal':
        if alpha == 'reweight':
            # print("hi")
            alpha = reweight(cls_num_list, beta=0.9999)
            # print(alpha)
        crit = FocalLoss(gamma=gamma, weight=alpha, device=device)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    # Define parameter groups
    if stage == 1:
        # Freeze backbone for Stage 1
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        optim_params = model.classifier.parameters() # Only train the head
    elif stage == 2:
        # Unfreeze the backbone for Stage 2
        for param in model.parameters():
            param.requires_grad = True
        optim_params = [
            {"params": model.classifier.parameters(), "lr": lr_head},
            {"params": model.base_model.parameters(), "lr": lr_backbone},
        ]
    else:
        raise ValueError(f"Unknown training stage: {stage}")
    
    # Initialize optimizer
    if optimizer == 'sgd':
        optim = SGD(
            optim_params,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer == 'adam':
        optim = torch.optim.Adam(
            optim_params,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    return crit, optim
