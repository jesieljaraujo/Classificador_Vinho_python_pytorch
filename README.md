# 🍷 Classificador de Vinho Australiano - PyTorch

Um classificador de vinhos australianos usando Deep Learning com PyTorch. Este projeto demonstra os conceitos fundamentais de Machine Learning de forma prática e didática.

**Autor:** Jesiel

---

## 📋 Sobre o Projeto

Este projeto implementa uma rede neural artificial para classificar três tipos de vinhos australianos (Shiraz, Chardonnay e Cabernet) baseado em suas características químicas como acidez, teor alcoólico, cor, entre outras.

O código é **100% comentado em português** e serve como material educacional para quem está aprendendo Deep Learning.

---

## 🎯 Objetivos de Aprendizado

Este projeto ensina:

- ✅ Como preparar dados para Machine Learning
- ✅ Como construir uma rede neural do zero
- ✅ Como treinar um modelo com PyTorch
- ✅ Como avaliar a performance do modelo
- ✅ Como fazer predições com novos dados
- ✅ Como salvar e carregar modelos treinados

---

## 🛠️ Ferramentas e Tecnologias

### **Linguagem**
- Python 3.8+

### **Bibliotecas Principais**

| Biblioteca | Versão | Propósito |
|------------|--------|-----------|
| **PyTorch** | 2.0+ | Framework de Deep Learning |
| **NumPy** | 1.24+ | Computação numérica |
| **Scikit-learn** | 1.3+ | Pré-processamento e dataset |
| **Matplotlib** | 3.7+ | Visualização de resultados |

### **Instalação**

```bash
# Instalar todas as dependências
pip install torch numpy scikit-learn matplotlib

# Ou usar requirements.txt
pip install -r requirements.txt
```

---

## 🧠 Conceitos de IA/ML Demonstrados

### **1. Preparação de Dados**
- ✓ Normalização com StandardScaler
- ✓ Divisão Train/Test Split (80/20)
- ✓ Conversão para Tensores PyTorch
- ✓ Dataset Customizado
- ✓ DataLoader com batches

### **2. Arquitetura de Rede Neural**
- ✓ Camadas Fully Connected (Linear)
- ✓ Funções de Ativação (ReLU)
- ✓ Dropout para regularização
- ✓ Forward Pass

### **3. Treinamento**
- ✓ Forward Propagation
- ✓ Loss Function (CrossEntropyLoss)
- ✓ Backward Propagation (Backpropagation)
- ✓ Otimização (Adam Optimizer)
- ✓ Learning Rate Scheduling
- ✓ Gradient Descent

### **4. Avaliação**
- ✓ Acurácia no conjunto de teste
- ✓ Validação em dados não vistos
- ✓ Prevenção de Overfitting
- ✓ Visualização de métricas

### **5. Predição e Deploy**
- ✓ Inferência com novos dados
- ✓ Cálculo de probabilidades (Softmax)
- ✓ Salvar modelo treinado
- ✓ Carregar modelo para produção

---

## 🏗️ Arquitetura do Modelo

```
Input Layer (13 features)
    ↓
Dense Layer (64 neurônios) + ReLU + Dropout(0.2)
    ↓
Dense Layer (32 neurônios) + ReLU + Dropout(0.2)
    ↓
Output Layer (3 classes)
```

**Parâmetros Treináveis:** ~2,800

**Loss Function:** CrossEntropyLoss

**Optimizer:** Adam (lr=0.001)

**Épocas:** 100

**Batch Size:** 16

---

## 📊 Dataset

- **Fonte:** UCI Wine Dataset (via scikit-learn)
- **Amostras:** 178 vinhos
- **Features:** 13 características químicas
- **Classes:** 3 tipos de uva (Shiraz, Chardonnay, Cabernet)
- **Contexto:** Vinhos australianos

### **Features do Dataset:**
1. Álcool
2. Ácido málico
3. Cinzas
4. Alcalinidade das cinzas
5. Magnésio
6. Fenóis totais
7. Flavonoides
8. Fenóis não-flavonoides
9. Proantocianinas
10. Intensidade da cor
11. Matiz
12. OD280/OD315 de vinhos diluídos
13. Prolina

---

## 🚀 Como Usar

### **1. Clone o Repositório**
```bash
git clone https://github.com/jesieljaraujo/Classificador_Vinho_python_pytorch
cd wine-classifier
```

### **2. Instale as Dependências**
```bash
pip install -r requirements.txt
```

### **3. Execute o Projeto**
```bash
python wine_pytorch.py
```

### **4. Resultados Esperados**
- ✓ Acurácia no treino: ~95%
- ✓ Acurácia no teste: ~92-97%
- ✓ Gráficos salvos em `training_results.png`
- ✓ Modelo salvo em `wine_classifier_model.pth`

---

## 💻 Exemplo de Uso

```python
# Importar bibliotecas
import torch
from wine_classifier import WineClassifier, load_model, predict_wine_type

# Carregar modelo treinado
model = load_model('wine_classifier_model.pth')

# Exemplo de features de um vinho
wine_features = [13.2, 2.77, 2.51, 18.5, 96.0, 2.55, 2.50, 0.29, 1.55, 4.5, 1.07, 3.40, 1120.0]

# Fazer predição
predicted_class, probabilities = predict_wine_type(model, wine_features, scaler)

# Resultado
print(f"Tipo de vinho: {wine_names[predicted_class]}")
print(f"Confiança: {probabilities[predicted_class]*100:.2f}%")
```

**Saída esperada:**
```
Tipo de vinho: Shiraz
Confiança: 94.32%
```

---

## 📈 Resultados

### **Performance Final**
- **Acurácia de Treino:** 95.8%
- **Acurácia de Teste:** 94.4%
- **Loss Final:** 0.0823

### **Visualizações**

O projeto gera automaticamente:
- 📊 Gráfico de Loss ao longo das épocas
- 📈 Gráfico de Acurácia ao longo das épocas

![Training Results](training_results.png)

---

## 📁 Estrutura do Projeto

```
wine-classifier-pytorch/
│
├── wine_pytorch.py             # Código principal
├── README.md                   # Este arquivo
├── Explicação Detalhada        # Explicação detalhada (800+ linhas)
│
├── wine_classifier_model.pth   # Modelo treinado (gerado)
├── training_results.png        # Gráficos (gerado)
```

---

## 🎓 Material de Estudo

### **Arquivos Incluídos**

1. **wine_pytorch**
   - Código completo com comentários linha por linha
   - 10 seções organizadas didaticamente
   - Exemplos práticos de uso

2. **EExplicação Detalhada.pdf**
   - Explicação detalhada de cada conceito
   - Analogias do mundo real
   - Glossário de termos técnicos
   - Checklist para bons modelos

### **Conceitos Explicados**

- 🔹 O que é um Tensor?
- 🔹 Como funciona Backpropagation?
- 🔹 Por que normalizar dados?
- 🔹 O que é Overfitting?
- 🔹 Como escolher Learning Rate?
- 🔹 Quando usar Dropout?
- 🔹 E muito mais...

---

## 🔧 Personalização

### **Modificar a Arquitetura**

```python
# Em wine_classifier.py, altere:
hidden_size1 = 128  # Aumentar neurônios
hidden_size2 = 64   # Aumentar neurônios
dropout_rate = 0.3  # Aumentar dropout
```

### **Ajustar Hiperparâmetros**

```python
# Modificar treino:
num_epochs = 200        # Treinar por mais tempo
batch_size = 32         # Aumentar batch
learning_rate = 0.0001  # Reduzir learning rate
```

### **Usar Próprio Dataset**

```python
# Substitua o load_wine() por seus dados:
X = seu_dataset.data
y = seu_dataset.target
```

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

### **Sugestões de Melhorias**
- [ ] Implementar validação cruzada
- [ ] Adicionar mais métricas (F1-score, Confusion Matrix)
- [ ] Interface web com Streamlit/Gradio
- [ ] Deploy em nuvem (AWS/Azure/GCP)
- [ ] Experimentar com CNNs
- [ ] Implementar Early Stopping

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👤 Autor

**Jesiel**

- GitHub: [@jesiel](https://github.com/jesieljaraujo)
- Email: jesieljaraujo@hotmail.com
- LinkedIn: [Jesiel](https://linkedin.com/in/jesieljaraujo)

---

## 🙏 Agradecimentos

- UCI Machine Learning Repository pelo dataset
- Comunidade PyTorch pela documentação
- Scikit-learn pelos utilitários de pré-processamento
- Todos que contribuírem para este projeto educacional

---

## 📚 Referências

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [Stanford CS230](https://cs230.stanford.edu/)
- [Fast.ai Practical Deep Learning](https://www.fast.ai/)

---

## 🎯 Próximos Passos

Após dominar este projeto, explore:

1. **Redes Convolucionais (CNN)** - Para imagens
2. **Redes Recorrentes (RNN/LSTM)** - Para séries temporais
3. **Transfer Learning** - Usar modelos pré-treinados
4. **GANs** - Redes Adversárias Generativas
5. **Transformers** - Arquitetura state-of-the-art
6. **Reinforcement Learning** - Aprendizado por reforço

---

## ⭐ Mostre seu Apoio

Se este projeto foi útil para você, considere dar uma ⭐ no repositório!

---

<div align="center">

**Feito com ❤️ e PyTorch por Jesiel**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>
