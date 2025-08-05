import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

# Configurações
class Config:
    
    #TODO: Definir configurações

    pass 

# Dataset customizado
class PoseDataset(Dataset):
    def __init__(self, features_dir, seq_length=Config.SEQUENCE_LENGTH):
        self.features = []
        self.labels = []
        self.seq_length = seq_length
        
        # Carrega features e labels
        for class_idx, class_name in enumerate(['normal', 'assault']):
            class_dir = Path(features_dir) / class_name
            for video_dir in class_dir.iterdir():
                if video_dir.is_dir():
                    feature_file = video_dir / 'features.npy'
                    if feature_file.exists():
                        try:
                            # Carregar a matriz de features (n_frames x n_features)
                            feature_data = np.load(feature_file)
                            
                            # Verificar se os dados têm forma adequada
                            if len(feature_data.shape) == 2:
                                # Se o arquivo contém a matriz de sequência
                                n_frames, n_features = feature_data.shape
                                
                                # Processar a sequência em pedaços de tamanho seq_length
                                for i in range(0, n_frames - self.seq_length + 1, self.seq_length // 2):  # Sobreposição de 50%
                                    seq = feature_data[i:i+self.seq_length]
                                    if len(seq) == self.seq_length:  # Garantir que temos uma sequência completa
                                        self.features.append(seq)
                                        self.labels.append(class_idx)
                            elif isinstance(feature_data, dict):
                                # Compatibilidade com o formato antigo (dicionário)
                                print(f"Formato antigo detectado em {feature_file}, convertendo...")
                                feature_keys = sorted(feature_data.keys())
                                n_frames = len(feature_data[feature_keys[0]])
                                
                                # Criar matriz de sequência
                                seq_matrix = np.zeros((n_frames, len(feature_keys)))
                                for i, key in enumerate(feature_keys):
                                    seq_matrix[:, i] = feature_data[key]
                                
                                # Processar a sequência em pedaços
                                for i in range(0, n_frames - self.seq_length + 1, self.seq_length // 2):
                                    seq = seq_matrix[i:i+self.seq_length]
                                    if len(seq) == self.seq_length:
                                        self.features.append(seq)
                                        self.labels.append(class_idx)
                        except Exception as e:
                            print(f"Erro ao carregar {feature_file}: {e}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]), 
            torch.LongTensor([self.labels[idx]])
        )

# Modelo RNN
class PoseRNN(nn.Module):
    def __init__(self, input_size, rnn_type='lstm'):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        
        # Camada RNN
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=Config.HIDDEN_SIZE,
                num_layers=Config.NUM_LAYERS,
                batch_first=True,
                dropout=Config.DROPOUT
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=Config.HIDDEN_SIZE,
                num_layers=Config.NUM_LAYERS,
                batch_first=True,
                dropout=Config.DROPOUT
            )
        else:  # RNN simples
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=Config.HIDDEN_SIZE,
                num_layers=Config.NUM_LAYERS,
                batch_first=True,
                dropout=Config.DROPOUT
            )
        
        # Camada de classificação
        self.fc = nn.Linear(Config.HIDDEN_SIZE, 2)  # 2 classes (normal, assault)
        self.dropout = nn.Dropout(Config.DROPOUT)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Pega apenas o último timestep
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Função de treino
def train_model(model, train_loader, val_loader, optimizer, criterion):
    best_acc = 0
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0
        
        # Loop de treino
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Validação
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        train_loss = running_loss / len(train_loader)
        
        # Salvar métricas
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Salvar melhor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_{model.rnn_type}_model.pth')
        
        print(f'Epoch {epoch+1}/{Config.EPOCHS} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.2%}')

    return train_losses, val_losses, val_accs

# Função de avaliação
def evaluate(model, loader, criterion):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()
    
    return running_loss / len(loader), correct / total

# Pipeline principal
def main():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # Carregar dados
    train_data = PoseDataset('data/splits/train')
    val_data = PoseDataset('data/splits/val')
    test_data = PoseDataset('data/splits/test')
    
    # Criar DataLoaders
    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE)
    
    # Verificar tamanho da entrada
    sample_input = train_data[0][0]
    input_size = sample_input.shape[-1]  # (seq_len, input_size)
    
    # Testar diferentes modelos
    for rnn_type in ['lstm', 'gru', 'rnn']:
        print(f'\n=== Treinando {rnn_type.upper()} ===')
        
        # Criar modelo
        model = PoseRNN(input_size, rnn_type=rnn_type).to(Config.DEVICE)
        
        # Loss e otimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        
        # Treinar
        train_losses, val_losses, val_accs = train_model(
            model, train_loader, val_loader, optimizer, criterion)
        
        # Plotar resultados
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.title(f'{rnn_type.upper()} Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accs, label='Validation Accuracy')
        plt.title(f'{rnn_type.upper()} Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Avaliar no teste
        print(f'\nAvaliação do {rnn_type.upper()} no conjunto de teste:')
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f'Test Accuracy: {test_acc:.2%}')

if __name__ == '__main__':
    main()