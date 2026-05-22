# Aplicação de Deep Learning e Explainable Artificial Intelligence na Detecção de Pneumonia em Radiografias Torácicas Utilizando DenseNet121 e Grad-CAM


O projeto propõe uma abordagem baseada em *Deep Learning* combinada com ferramentas de *Explainable Artificial Intelligence* (XAI) para a triagem automatizada, auditável e clinicamente explicável de pneumonia a partir de imagens digitais de raios-X de tórax.

O objetivo central deste trabalho é mitigar o problema da "caixa-preta" (*black-box problem*) intrínseco às redes neurais convolucionais profundas. Ao integrar o algoritmo **Grad-CAM**, o sistema não apenas emite um diagnóstico preditivo, mas também gera mapas de ativação que evidenciam visualmente quais regiões do parênquima pulmonar justificaram a tomada de decisão do modelo, promovendo a transparência e a confiança em ambientes de saúde digital.

---

## 📑 Contextualização Clínica e Fundamentação Teórica

### O Desafio Clínico da Pneumonia
A pneumonia permanece consolidada como uma das patologias infecciosas de maior impacto na saúde pública global, figurando de forma persistente entre as principais causas de morbidade e mortalidade mundial. Acometendo o parênquima pulmonar e os espaços alveolares, o preenchimento por exsudato inflamatório compromete as trocas gasosas. A agilidade no diagnóstico e o início imediato da terapêutica adequada são os pilares fundamentais para mitigar desfechos fatais. O raio-X de tórax destaca-se como o exame de triagem de primeira linha devido ao seu baixo custo e ampla disponibilidade, embora a interpretação esteja sujeita a variações interobservador substanciais.

### Por que a Arquitetura DenseNet121?
A escolha da **DenseNet121** (*Densely Connected Convolutional Networks*) baseia-se em sua eficiência no reaproveitamento de características. Diferente das arquiteturas convolucionais tradicionais (como ResNet ou VGG), as camadas de uma DenseNet são conectadas diretamente a todas as camadas subsequentes. 
* **Conexões Densas:** Cada camada recebe como entrada os mapas de recursos (*feature maps*) de todas as camadas anteriores e passa seus próprios mapas para as seguintes.
* **Vantagens:** Isso mitiga o problema do desvanecimento do gradiente (*vanishing gradient*), reduz drasticamente o número de parâmetros operacionais (tornando o modelo menos propenso ao *overfitting*) e incentiva a reutilização de características de baixo e alto nível em todo o fluxo da rede.

### A Necessidade de Explicabilidade (Grad-CAM)
Modelos de alta acurácia são frequentemente descartados na prática clínica devido à falta de interpretabilidade. O **Grad-CAM** (*Gradient-weighted Class Activation Mapping*) resolve este impasse utilizando os gradientes de qualquer classe-alvo, fluindo para a última camada convolucional relevante, para produzir um mapa de localização grosseiro que destaca as regiões importantes na imagem. Clinicamente, isso permite correlacionar as densidades focais apontadas pela IA diretamente com os limites geométricos dos infiltrados alveolares e opacidades pulmonares.

---

## 📊 Estrutura e Pipeline do Código Python

O pipeline foi construído utilizando o **TensorFlow / Keras 3** e está estruturado em um fluxo ponta a ponta:

### 1. Pré-processamento e Data Augmentation
Para lidar com a variabilidade inerente aos dispositivos de raios-X e evitar o sobreajuste, as matrizes de entrada são redimensionadas para **$224 \times 224 \times 3$** com lotes operacionais (*batch size*) de **16**. O `ImageDataGenerator` aplica transformações geométricas controladas (como rotações pontuais, zoom dinâmico e espelhamento horizontal) apenas no conjunto de treino.

### 2. Customização da Rede e Estratégia de Treinamento
O modelo final é composto pelo extrator de características congelado da DenseNet121 (pré-treinado na ImageNet) acoplado a uma cabeça de classificação densa customizada:
* Camada `GlobalAveragePooling2D` para redução dimensional.
* Camada de Regularização `Dropout(0.5)`.
* Camada Intermediária `Dense` com 128 neurônios e ativação **ReLU**.
* Camada de Regularização `Dropout(0.3)`.
* Camada de Saída `Dense` com 1 neurônio e função de ativação **Sigmoide** (adequada para classificação binária: `NORMAL` vs. `PNEUMONIA`).

### 3. Dinâmica de Otimização (Two-Stage Training)
* **Fase 1 (Transfer Learning):** Congelamento total da base e treinamento apenas das camadas densas superiores com otimizador **Adam** e taxa de aprendizado (*Learning Rate*) inicial de $10^{-4}$ por 10 épocas.
* **Fase 2 (Fine-Tuning):** Descongelamento seletivo das **últimas 50 camadas** da DenseNet121, aplicando uma taxa de aprendizado reduzida ($10^{-5}$) para refinar os filtros convolucionais profundos sem destruir os pesos de baixo nível já consolidados. Callbacks como `ReduceLROnPlateau` e `EarlyStopping` monitoram a perda de validação.

---

## 📈 Resultados, Métricas de Desempenho e Discussão

A avaliação do modelo foi conduzida de forma rigorosa utilizando o conjunto de dados de teste (completamente isolado durante as etapas de treinamento e fine-tuning). Os resultados foram estratificados para mitigar os riscos de falsos negativos, que possuem o maior custo clínico no cenário de triagem pneumônica.

### 1. Desempenho Preditivo (Classification Report)

O modelo alcançou alta robustez global, destacando-se na métrica de **Sensibilidade (*Recall*)** para a classe `PNEUMONIA`, garantindo que a grande maioria dos pacientes afetados seja corretamente identificada na triagem inicial.

| Classe | Precisão (*Precision*) | Sensibilidade (*Recall*) | F1-Score | Suporte (Imagens) |
| :--- | :---: | :---: | :---: | :---: |
| **NORMAL** (Saudável) | 0.94 | 0.85 | 0.89 | 234 |
| **PNEUMONIA** (Patológico) | 0.91 | 0.97 | 0.94 | 390 |
| **Média Global (Macro Avg)** | 0.93 | 0.91 | 0.92 | 624 |
| **Média Ponderada (Weighted Avg)** | 0.92 | 0.92 | 0.92 | 624 |

#### Análise das Métricas:
* **Alta Sensibilidade (0.97):** Significa que o sistema minimiza os alarmes falsos de saúde (falsos negativos), identificando 97% dos casos reais de pneumonia.
* **Área Abaixo da Curva (AUC-ROC):** O pipeline consolidou uma pontuação de **AUC de 0.963**, demonstrando excelente capacidade de discriminação estatística entre as distribuições das duas classes.

---

### 2. Matriz de Confusão

A matriz de confusão abaixo detalha o comportamento das predições absolutas do modelo frente aos rótulos reais estabelecidos pelos especialistas médicos médicos (*ground truth*):

| | Predito: NORMAL | Predito: PNEUMONIA |
| :--- | :---: | :---: |
| **Real: NORMAL** | **199** *(Verdadeiros Negativos)* | **35** *(Falsos Positivos)* |
| **Real: PNEUMONIA** | **11** *(Falsos Negativos)* | **379** *(Verdadeiros Positivos)* |

#### Discussão dos Erros Comportamentais:
*Os 35 casos de falsos positivos comumente correlacionam-se a radiografias com artefatos técnicos de expiração incompleta ou proeminências vasculares normais que simulam opacidades biológicas. Já a taxa marginal de falsos negativos (11 casos) foi empurrada ao mínimo viável através das funções de perda ponderadas e otimização de limiar diagnóstica.*

---

### 3. Curvas de Aprendizado e Convergência (Plots do Matplotlib)

O comportamento do histórico de treinamento e fine-tuning foi exportado de forma gráfica e pode ser visualizado abaixo. O ponto de transição (Época 10) demarca o momento em que as últimas 50 camadas foram descongeladas:

```markdown
![Curvas de Convergência - Acurácia e Perda](outputs/learning_curves.png)

![Inspeção Visual Grad-CAM](outputs/gradcam_result.png)

chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

DATASET_PATH = r"C:/SeuCaminho/Para/O/Dataset/chest_xray"
