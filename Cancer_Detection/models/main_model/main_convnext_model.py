import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from convnet_one import Creating_Convnet
from ..evaluation_extended import ExtendedEvaluation, plot_loss_curves


def default_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FullTrainingOfModel(Creating_Convnet):
    """
    Extends Creating_Convnet to train multiple models, save/load weights,
    and perform a full suite of evaluations including loss curves,
    confusion matrices, ROC curves, and reporting for both validation and test sets.
    """
    def __init__(self, train_dataset, val_dataset, test_dataset,
                 batch_size=32, lr=1e-4, num_epochs=10, device=None, class_names=None):
        self.device = device or default_device()
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        self.lr = lr
        self.num_epochs = num_epochs
        self.class_names = class_names

        # Storage for models and histories
        self.models = {}
        self.histories = {}
        self.eval_results = {}

    def initialize_models(self, num_models):
        """Instantiate multiple ConvNet models with optimizers and loss criteria."""
        for i in range(num_models):
            model = self.creating_single_model()
            model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()

            self.models[f'model_{i}'] = model
            self.histories[f'model_{i}'] = {
                'optimizer': optimizer,
                'criterion': criterion,
                'train_loss': [],
                'val_loss': []
            }

        if not self.models:
            raise ValueError("No models were initialized. Call initialize_models first.")

    def train(self):
        """Train each model, logging training and validation losses per epoch."""
        for name, model in self.models.items():
            history = self.histories[name]
            optimizer = history['optimizer']
            criterion = history['criterion']

            for epoch in range(1, self.num_epochs + 1):
                model.train()
                running_loss = 0.0
                for images, labels in self.train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * images.size(0)
                train_epoch_loss = running_loss / len(self.train_loader.dataset)
                history['train_loss'].append(train_epoch_loss)

                # Validation loss
                model.eval()
                val_running = 0.0
                with torch.no_grad():
                    for images, labels in self.val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_running += loss.item() * images.size(0)
                val_epoch_loss = val_running / len(self.val_loader.dataset)
                history['val_loss'].append(val_epoch_loss)

                print(f"{name} | Epoch {epoch}/{self.num_epochs} "
                      f"Train Loss: {train_epoch_loss:.4f} Val Loss: {val_epoch_loss:.4f}")

    def save_model(self, name, filepath):
        """Save a single model's state_dict to the given filepath."""
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.models[name].state_dict(), filepath)
        print(f"Saved {name} weights to {filepath}")

    def save_all_models(self, directory):
        """Save all models' weights into the specified directory."""
        os.makedirs(directory, exist_ok=True)
        for name in self.models:
            path = os.path.join(directory, f"{name}_weights.pth")
            self.save_model(name, path)

    def load_model(self, name, filepath):
        """Load weights into a model from the given filepath."""
        if name not in self.models:
            # Initialize if missing
            model = self.creating_single_model()
            model.to(self.device)
            self.models[name] = model
        self.models[name].load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Loaded {name} weights from {filepath}")

    def train_and_save(self, save_dir):
        """Convenience method: trains all models then saves their weights."""
        self.train()
        self.save_all_models(save_dir)

    def evaluate_models(self):
        """Run ExtendedEvaluation on validation and test sets for each model and store results, returning metrics dict."""
        for name, model in self.models.items():
            results = {}
            for split, loader in [('val', self.val_loader), ('test', self.test_loader)]:
                evaluator = ExtendedEvaluation(model, loader,
                                               device=self.device,
                                               class_names=self.class_names)
                metrics = evaluator.compute_all()
                results[split] = metrics
                # Plot confusion matrix
                fig_cm = evaluator.plot_confusion_matrix(normalize=False)
                fig_cm.suptitle(f"{name} - {split} Confusion Matrix")
                fig_cm.show()
                # Plot normalized confusion matrix
                fig_cm_norm = evaluator.plot_confusion_matrix(normalize=True)
                fig_cm_norm.suptitle(f"{name} - {split} Confusion Matrix (Normalized)")
                fig_cm_norm.show()
                # Plot ROC curves
                fig_roc = evaluator.plot_roc_curve()
                fig_roc.suptitle(f"{name} - {split} ROC Curves")
                fig_roc.show()

            self.eval_results[name] = results
        return self.eval_results

    def plot_all_loss_curves(self):
        """Plot training and validation loss curves for all models."""
        for name, history in self.histories.items():
            fig = plot_loss_curves(history['train_loss'], history['val_loss'])
            fig.suptitle(f"{name} Loss Curves over Epochs")
            fig.show()

# Example usage:
# trainer = FullTrainingOfModel(train_ds, val_ds, test_ds, num_epochs=15, class_names=['benign','malignant'])
# trainer.initialize_models(num_models=3)
# trainer.train_and_save('model_weights/')
# # Later, to load:
# trainer = FullTrainingOfModel(train_ds, val_ds, test_ds, class_names=['benign','malignant'])
# trainer.load_model('model_0', 'model_weights/model_0_weights.pth')
# predictions = trainer.models['model_0'](some_input)
