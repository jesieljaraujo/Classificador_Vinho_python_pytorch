"""
Classificador de Vinhos Australianos usando PyTorch
Este projeto demonstra os conceitos fundamentais de Machine Learning:
- Redes Neurais Artificiais
- Treinamento com Backpropagation
- Avaliação de Modelos
- Predições
"""

import torch
import torch.nn as nn  # Neural Network modules
import torch.optim as optim  # Optimization algorithms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.datasets import load_wine  # Dataset de vinhos
from sklearn.model_selection import train_test_split  # Divisão treino/teste
from sklearn.preprocessing import StandardScaler  # Normalização
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PREPARAÇÃO DOS DADOS (Data Preparation)
# ==============================================================================

print("=" * 60)
print("CLASSIFICADOR DE VINHOS AUSTRALIANOS - PyTorch")
print("=" * 60)

# Carregar dataset de vinhos (3 tipos de uva: Shiraz, Chardonnay, Cabernet)
wine_data = load_wine()
X = wine_data.data  # Features: acidez, álcool, cor, etc.
y = wine_data.target  # Labels: 0, 1, 2 (tipos de vinho)

print(f"\n📊 Dataset carregado:")
print(f"   - Amostras: {X.shape[0]}")
print(f"   - Features: {X.shape[1]}")
print(f"   - Classes: {len(np.unique(y))}")
print(f"   - Nomes das classes: {wine_data.target_names}")

# Dividir dados em treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% para teste
    random_state=42,  # Seed para reprodutibilidade
    stratify=y  # Manter proporção das classes
)

# Normalizar os dados (média=0, desvio padrão=1)
# Isso ajuda o modelo a convergir mais rápido
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit no treino
X_test = scaler.transform(X_test)  # Apenas transform no teste

# Converter para tensores PyTorch (estrutura de dados do PyTorch)
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

print(f"\n✓ Dados normalizados e convertidos para tensores PyTorch")

# ==============================================================================
# 2. CRIAÇÃO DO DATASET CUSTOMIZADO (Custom Dataset)
# ==============================================================================

class WineDataset(Dataset):
    """
    Dataset customizado para carregar dados de vinho
    PyTorch usa essa classe para gerenciar dados durante o treinamento
    """
    def __init__(self, X, y):
        # Construtor: inicializa o dataset
        self.X = X
        self.y = y
    
    def __len__(self):
        # Retorna o tamanho do dataset
        return len(self.X)
    
    def __getitem__(self, idx):
        # Retorna um item (amostra) pelo índice
        return self.X[idx], self.y[idx]

# Criar datasets
train_dataset = WineDataset(X_train_tensor, y_train_tensor)
test_dataset = WineDataset(X_test_tensor, y_test_tensor)

# DataLoader: carrega dados em lotes (batches) durante o treinamento
train_loader = DataLoader(
    train_dataset, 
    batch_size=16,  # Processa 16 amostras por vez
    shuffle=True  # Embaralha os dados a cada época
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=16, 
    shuffle=False  # Não embaralha os dados de teste
)

print(f"✓ DataLoaders criados (batch_size=16)")

# ==============================================================================
# 3. DEFINIÇÃO DO MODELO (Neural Network Architecture)
# ==============================================================================

class WineClassifier(nn.Module):
    """
    Rede Neural Feedforward para classificação de vinhos
    Arquitetura: Input -> Hidden Layer 1 -> Hidden Layer 2 -> Output
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        # Chamar construtor da classe pai
        super(WineClassifier, self).__init__()
        
        # Camada 1: Input -> Hidden Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size1)
        
        # Camada 2: Hidden Layer 1 -> Hidden Layer 2
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        
        # Camada 3: Hidden Layer 2 -> Output
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
        # Função de ativação ReLU (Rectified Linear Unit)
        # ReLU(x) = max(0, x) - Introduz não-linearidade
        self.relu = nn.ReLU()
        
        # Dropout: desliga aleatoriamente 20% dos neurônios durante treino
        # Isso previne overfitting (memorização dos dados)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """
        Forward pass: define como os dados fluem pela rede
        """
        # Passa pela primeira camada e aplica ReLU
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Passa pela segunda camada e aplica ReLU
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Camada de saída (sem ativação aqui - será feita no loss)
        x = self.fc3(x)
        
        return x

# Instanciar o modelo
input_size = X_train.shape[1]  # 13 features
hidden_size1 = 64  # 64 neurônios na primeira camada oculta
hidden_size2 = 32  # 32 neurônios na segunda camada oculta
num_classes = 3  # 3 tipos de vinho

model = WineClassifier(input_size, hidden_size1, hidden_size2, num_classes)

print(f"\n🧠 Modelo criado:")
print(f"   - Arquitetura: {input_size} -> {hidden_size1} -> {hidden_size2} -> {num_classes}")
print(f"   - Parâmetros treináveis: {sum(p.numel() for p in model.parameters())}")

# ==============================================================================
# 4. CONFIGURAÇÃO DO TREINAMENTO (Training Setup)
# ==============================================================================

# Função de perda (Loss Function)
# CrossEntropyLoss: ideal para classificação multiclasse
criterion = nn.CrossEntropyLoss()

# Otimizador (Optimizer)
# Adam: algoritmo de otimização adaptativo (versão melhorada do SGD)
optimizer = optim.Adam(
    model.parameters(),  # Parâmetros do modelo a serem otimizados
    lr=0.001  # Learning rate (taxa de aprendizado)
)

# Scheduler: reduz o learning rate ao longo do tempo
# Isso ajuda o modelo a convergir melhor
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',  # Reduz quando a loss para de diminuir
    patience=5,  # Aguarda 5 épocas antes de reduzir
    factor=0.5  # Reduz pela metade
)

print(f"\n⚙️  Configuração do treinamento:")
print(f"   - Loss Function: CrossEntropyLoss")
print(f"   - Optimizer: Adam (lr=0.001)")
print(f"   - Scheduler: ReduceLROnPlateau")

# ==============================================================================
# 5. TREINAMENTO DO MODELO (Model Training)
# ==============================================================================

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """
    Função para treinar o modelo
    """
    # Listas para armazenar métricas
    train_losses = []
    train_accuracies = []
    
    print(f"\n🚀 Iniciando treinamento por {num_epochs} épocas...\n")
    
    for epoch in range(num_epochs):
        # Modo de treinamento (ativa dropout e outras camadas específicas)
        model.train()
        
        running_loss = 0.0  # Acumula a loss da época
        correct = 0  # Conta predições corretas
        total = 0  # Conta total de amostras
        
        # Iterar sobre os batches
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Forward pass: calcular predições
            outputs = model(data)
            
            # Calcular loss (erro entre predição e valor real)
            loss = criterion(outputs, targets)
            
            # Backward pass: calcular gradientes
            optimizer.zero_grad()  # Zerar gradientes anteriores
            loss.backward()  # Calcular novos gradientes (backpropagation)
            
            # Atualizar pesos do modelo
            optimizer.step()
            
            # Estatísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # Pegar classe com maior probabilidade
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        # Calcular métricas da época
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Atualizar learning rate
        scheduler.step(epoch_loss)
        
        # Imprimir progresso a cada 10 épocas
        if (epoch + 1) % 10 == 0:
            print(f"Época [{epoch+1}/{num_epochs}] | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Acurácia: {epoch_accuracy:.2f}%")
    
    print(f"\n✓ Treinamento concluído!")
    return train_losses, train_accuracies

# Treinar o modelo por 100 épocas
num_epochs = 100
train_losses, train_accuracies = train_model(
    model, train_loader, criterion, optimizer, num_epochs
)

# ==============================================================================
# 6. AVALIAÇÃO DO MODELO (Model Evaluation)
# ==============================================================================

def evaluate_model(model, test_loader):
    """
    Avaliar o modelo no conjunto de teste
    """
    # Modo de avaliação (desativa dropout)
    model.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    # Desabilitar cálculo de gradientes (economiza memória)
    with torch.no_grad():
        for data, targets in test_loader:
            # Forward pass
            outputs = model(data)
            
            # Obter predições
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            all_predictions.extend(predicted.numpy())
            all_targets.extend(targets.numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_predictions, all_targets

print(f"\n📈 Avaliando modelo no conjunto de teste...")
test_accuracy, predictions, true_labels = evaluate_model(model, test_loader)

print(f"\n{'='*60}")
print(f"RESULTADOS FINAIS")
print(f"{'='*60}")
print(f"Acurácia no conjunto de TESTE: {test_accuracy:.2f}%")
print(f"{'='*60}")

# ==============================================================================
# 7. VISUALIZAÇÃO DOS RESULTADOS (Results Visualization)
# ==============================================================================

# Criar gráfico de loss e acurácia
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Gráfico 1: Loss ao longo das épocas
ax1.plot(train_losses, label='Training Loss', color='red', linewidth=2)
ax1.set_xlabel('Época', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss durante o Treinamento', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Gráfico 2: Acurácia ao longo das épocas
ax2.plot(train_accuracies, label='Training Accuracy', color='green', linewidth=2)
ax2.set_xlabel('Época', fontsize=12)
ax2.set_ylabel('Acurácia (%)', fontsize=12)
ax2.set_title('Acurácia durante o Treinamento', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Gráficos salvos em 'training_results.png'")

# ==============================================================================
# 8. FAZER PREDIÇÕES (Making Predictions)
# ==============================================================================

def predict_wine_type(model, features, scaler, wine_names):
    """
    Fazer predição para uma nova amostra de vinho
    """
    # Modo de avaliação
    model.eval()
    
    # Normalizar features
    features_scaled = scaler.transform([features])
    
    # Converter para tensor
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Fazer predição
    with torch.no_grad():
        output = model(features_tensor)
        
        # Aplicar softmax para obter probabilidades
        probabilities = torch.softmax(output, dim=1)
        
        # Obter classe predita
        _, predicted_class = torch.max(output, 1)
    
    return predicted_class.item(), probabilities[0].numpy()

# Exemplo de predição com uma nova amostra
print(f"\n🍷 Exemplo de Predição:")
print(f"{'='*60}")

# Usar a primeira amostra do conjunto de teste
sample_features = X_test[0]
true_class = y_test[0]

predicted_class, probabilities = predict_wine_type(
    model, sample_features, scaler, wine_data.target_names
)

print(f"Amostra: {sample_features[:3]}... (primeiras 3 features)")
print(f"\nClasse Real: {wine_data.target_names[true_class]}")
print(f"Classe Predita: {wine_data.target_names[predicted_class]}")
print(f"\nProbabilidades:")
for i, prob in enumerate(probabilities):
    print(f"  - {wine_data.target_names[i]}: {prob*100:.2f}%")

# ==============================================================================
# 9. SALVAR O MODELO (Save Model)
# ==============================================================================

# Salvar modelo treinado
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'input_size': input_size,
    'hidden_size1': hidden_size1,
    'hidden_size2': hidden_size2,
    'num_classes': num_classes
}, 'wine_classifier_model.pth')

print(f"\n✓ Modelo salvo em 'wine_classifier_model.pth'")

# ==============================================================================
# 10. CARREGAR MODELO (Load Model) - Exemplo
# ==============================================================================

def load_model(filepath):
    """
    Carregar modelo salvo
    """
    checkpoint = torch.load(filepath)
    
    # Recriar modelo com mesma arquitetura
    loaded_model = WineClassifier(
        checkpoint['input_size'],
        checkpoint['hidden_size1'],
        checkpoint['hidden_size2'],
        checkpoint['num_classes']
    )
    
    # Carregar pesos
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()
    
    return loaded_model

print(f"\n✓ Para carregar o modelo: loaded_model = load_model('wine_classifier_model.pth')")

print(f"\n{'='*60}")
print(f"PROJETO CONCLUÍDO COM SUCESSO! 🎉")
print(f"{'='*60}\n")