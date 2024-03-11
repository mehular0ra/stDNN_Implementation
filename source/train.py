import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings("ignore")

import ipdb

class stDNNTrainer:
    def __init__(self, model, train_loader, val_loader=None, batch_size=32, learning_rate=0.0001, save_path='best_model.pth'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.best_val_loss = float('inf')
        self.save_path = save_path  # Path to save the best model
    
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.float(), labels.float()
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs.squeeze())
            preds = torch.round(probs)

            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(labels.cpu().detach().numpy())
            all_probs.extend(probs.cpu().detach().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = running_loss / len(self.train_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = np.nan  # Handle cases where AUC can't be computed
        
        return avg_loss, accuracy, precision, recall, f1, auc
    
    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, accuracy, precision, recall, f1, auc = self.train_epoch()

            print(f"Epoch {epoch+1}/{epochs}, Train - Loss: {train_loss:.4f}, Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            if self.val_loader:
                val_metrics = self.validate()
                print(f"Validation - Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
                print()
                # Save the best model based on validation loss
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"Saved new best model at epoch {epoch+1}")

    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.float(), labels.float()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs.squeeze())
                preds = torch.round(probs)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_val_loss = val_loss / len(self.val_loader)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        
        # Handle AUC calculation safely
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = np.nan  # Assign NaN if AUC can't be computed
        
        metrics = {
            "val_loss": avg_val_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }
        
        return metrics
    

def cross_validate_model(model_class, dataset, num_splits=5, batch_size=32, epochs=15, learning_rate=0.0001):
    kf = KFold(n_splits=num_splits, shuffle=True)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
        print(f"Fold {fold + 1}/{num_splits}")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        
        model = model_class(num_channels=246, num_classes=1)  # Ensure this matches your model's expected input
        trainer = stDNNTrainer(model, train_loader, val_loader, batch_size, learning_rate)
        
        trainer.train(epochs)
        metrics = trainer.validate()  # Collect metrics from validation
        
        print(f"Fold {fold+1} Metrics: {metrics}")  # Optionally print out fold metrics here for immediate feedback
        
        fold_results.append(metrics)
        
    return fold_results
