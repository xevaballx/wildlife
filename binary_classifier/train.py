from torch.optim import SGD
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from binary_classifier.utils import set_seeds
import torch

import wandb
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

def setup_training(
        model, criterion='cross_entropy', optimizer="sdg", 
        lr=0.001, momentum=0.9 ):
    """
    Set up the optimizer and loss function based on the training configuration.

    Returns:
        criterion, optimizer objects
    """
    if criterion == 'cross_entropy':
        crit = CrossEntropyLoss()
    elif(criterion == "binary_cross_entropy"):
        crit = BCEWithLogitsLoss()
        

    if optimizer == 'sgd':
        optim = SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
        )

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
    tracking_loss = []  # List to store loss for every batch

    for batch_n, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).long()

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        tracking_loss.append(loss.item())  # Store batch loss for tracking 

        # backward and optimize
        optimizer.zero_grad()
        loss.backward() # wrt
        optimizer.step()

        # ✨ W&B: Log loss over training steps, visualized in the UI live
        wandb.log({"loss" : loss, "epoch": epoch+1,})

        # Print progress every 10 batches
        if (batch_n + 1 ) % 10 == 0: 
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                .format(epoch+1, config["train"]["epochs"], batch_n+1, len(train_loader), loss.item()))
            
        if(device=="cuda"):
            torch.cuda.empty_cache()
            
    avg_loss = total_loss / len(train_loader)
    return avg_loss, tracking_loss

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
    log_labels = labels.cpu().numpy()
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

################


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
        "1present", "2not_present",
    ])
    
    with torch.no_grad():
        for batch_n, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            image_ids = batch["image_id"]

            # forward pass
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)  # For logging
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            print()
            print("#########################################################")
            print(f"Batch: {batch_n}")
            print(f"Predictions:")
            print(predicted)
            print()
            print(f"labels:")
            print(labels)
            print("#########################################################")
            print()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Store predictions and labels for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())  # Convert one-hot to class indices
            

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
            "animal", "not_animal"
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
        
    print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    
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


