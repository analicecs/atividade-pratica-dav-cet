# 🎯 Atividade Prática Detecção de Agressões

> **Objetivo:** Implementar um pipeline completo para detectar eventos em vídeos.

## 📁 Estrutura do Projeto

```
DETECTOR-DE-EVENTOS-EM-VIDEOS/
├── data/                          # 📂 Vídeos e datasets
├── src/                           # 📁 Código fonte principal
├── metrics/                       # 📈 Avaliação de performance
├── preprocessing/                 # 🔄 Módulo de pré-processamento
│   ├── __pycache__/
│   ├── dataProcessor.py          # Processamento de vídeos e frames
│   ├── features_data.py          # Dados de características
│   ├── features.py               # Extração de características
│   ├── keypoints.py              # Detecção de pontos de interesse
│   ├── representation.py         # Representação de dados
│   ├── temporal_transformer.py   # Análise temporal
│   └── tracker_ds.py             # Rastreamento de objetos
├── training/                     # 🤖 Scripts de treinamento
│   ├── __pycache__/
│   ├── rnn_experiments.py        # Experimentos com RNN
│   └── train.py                  # Treinamento principal
├── .gitignore
├── app.py                        # Aplicação principal
├── best_gru_model.pth            # Modelo GRU treinado
├── best_lstm_model.pth           # Modelo LSTM treinado
├── best_rnn_model.pth            # Modelo RNN treinado
├── cookies.txt
├── get_cookies.py                # Coleta os cookies
├── README.md                     # Este guia
├── requirements.txt              # Dependências
└── yolov8n-pose.pt               # Modelo YOLOv8 para pose
```

---

## Atividades

### **Etapa 1: Pré-processamento (`src/preprocessing/dataProcessor.py`)**
**O que você vai implementar:**
- [] Função `extrair_frames()` - Extração de frames do vídeo
- [] Função `pre_processar_frame()` - Redimensionamento e normalização

**💡 Conceitos aplicados:** OpenCV básico, pré-processamento

---

### **Etapa 2: Detecção de Keypoints (`src/preprocessing/keypoints.py`)**
**O que você vai implementar:**
- [] Função `extract_keypoints_extended` - Usar o YOLO para detecção de pontos de interesse
- [] Função `draw_keypoints` - Usar OpenCV para desenhar os pontos de interesse

**💡 Conceitos aplicados:** YOLO, pose estimation, keypoint detection, OpenCV visualization

---

### **Etapa 3: Extração de Features (`src/preprocessing/features.py` e `src/preprocessing/representation.py`)**
**O que você vai implementar:**

`src/preprocessing/features.py`:
- [] Definir configurações de ângulos corporais importantes (`angle_configs`)
- [] Implementar normalização Min-Max (calcular `min_val`, `max_val` e aplicar fórmula)

`src/preprocessing/representation.py`:
- [] Função `calculate_velocity` - Cálculo da velocidade dos keypoints (primeira derivada)
- [] Função `calculate_acceleration` - Cálculo da aceleração dos keypoints (segunda derivada)  
- [] Função `calculate_angle` - Completar cálculo do produto escalar e ângulo

**💡 Conceitos aplicados:** Feature extraction, normalização Min-Max, cálculo de derivadas (velocidade/aceleração), geometria computacional, análise temporal

---

### **Etapa 4: Treinamento (`src/training/train.py`)**
**O que você vai implementar:**
```
class Config:
    MAX_SEQUENCE_LENGTH =     # Comprimento máximo das sequências
    HIDDEN_SIZE =             # Tamanho das camadas ocultas
    NUM_LAYERS =              # Número de camadas RNN
    DROPOUT =                 # Taxa de dropout
    LEARNING_RATE =           # Taxa de aprendizado
    BATCH_SIZE =              # Tamanho do lote
    EPOCHS =                  # Número de épocas
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED =                    # Semente para reprodutibilidade
```

---

**💡 Conceitos aplicados:** configuração de hiperparâmetros

---

## 🛠️ Setup do Ambiente

### **Pré-requisitos:**

#### **1. Verificar Python (versão 3.8+):**
```bash
python --version
# ou
python3 --version
```

**Se não tiver Python instalado:**
- **Windows:** Baixe em https://python.org/downloads
- **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install python3 python3-pip`
- **Linux (CentOS/RHEL):** `sudo yum install python3 python3-pip`

#### **2. Verificar pip:**
```bash
pip --version
# ou 
pip3 --version
```

**Se não tiver pip:**
```bash
# Linux
sudo apt install python3-pip

# Windows (geralmente já vem com Python)
python -m ensurepip --upgrade
```

---

### **Criação do Ambiente Virtual:**

#### **Linux/macOS:**
```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente
source venv/bin/activate

# Verificar se está ativo (deve aparecer (venv) no terminal)
which python
```

#### **Windows:**
```cmd
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente
venv\Scripts\activate

# Verificar se está ativo (deve aparecer (venv) no prompt)
where python
```

---

### **Instalação das Dependências:**
```bash
# Atualizar pip (recomendado)
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt

# Verificar instalação
pip list
```

---

### **Desativar Ambiente Virtual:**
```bash
deactivate
```

---

### **⚠️ Problemas Comuns:**

**Python não encontrado:**
- Windows: Adicionar Python ao PATH durante instalação
- Linux: Usar `python3` ao invés de `python`

**Erro de permissão no Linux:**
```bash
sudo chown -R $USER:$USER venv/
```

### **Requisitos:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Como Executar

### 1. Extraia keypoints (Opcional: Há keypoints em data/processed/keypoints/):
```bash
python3 -m src.preprocessing.keypoints
```

### 2. Extraia características (Opcional: Há features em data/processed/sequences/):
```bash
python3 -m src.preprocessing.features
```

### 3. Treine o modelo (Opcional: Há modelos treinados na raiz do projeto):
```bash
python3 -m src.training.train
```

### 4. Execute a aplicação:
```bash
streamlit run app.py
```

---

## 📊 Estrutura dos Dados

**Input:** Link de vídeos do YouTube  
**Processamento:** Frames → Keypoints → Features → Modelo  
**Output:** Classificação de eventos/ações

---

*Atividade prática desenvolvida para prática de visão computacional e machine learning aplicados.*
