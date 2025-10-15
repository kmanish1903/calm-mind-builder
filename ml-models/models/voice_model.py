import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

class VoiceMoodLSTM(nn.Module):
    """LSTM model for voice-based mood detection"""
    
    def __init__(self, input_size: int = 65, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

class VoiceMoodCNN(nn.Module):
    """CNN model for voice-based mood detection"""
    
    def __init__(self, input_size: int = 65, num_classes: int = 5):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions
        conv_output_size = input_size // 8 * 256  # After 3 pooling operations
        
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, input_size) -> (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class VoiceModelTrainer:
    """Trainer for voice mood detection models"""
    
    def __init__(self, model_type: str = 'lstm', device: str = 'cpu'):
        self.device = device
        self.model_type = model_type
        
    def create_model(self, input_size: int, num_classes: int = 5):
        """Create model based on type"""
        if self.model_type == 'lstm':
            model = VoiceMoodLSTM(input_size, num_classes=num_classes)
        else:
            model = VoiceMoodCNN(input_size, num_classes=num_classes)
        
        return model.to(self.device)
    
    def train_model(self, model, train_loader, val_loader, epochs: int = 50):
        """Train the voice mood model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        best_val_acc = 0
        train_losses, val_losses = [], []
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                if self.model_type == 'lstm':
                    # Reshape for LSTM: (batch, seq_len, features)
                    batch_features = batch_features.unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    if self.model_type == 'lstm':
                        batch_features = batch_features.unsqueeze(1)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            val_acc = correct / total
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_voice_{self.model_type}_model.pth')
            
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss / len(val_loader))
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.4f}')
        
        return train_losses, val_losses, best_val_acc