# Aplicação de Deep Learning e Explainable Artificial Intelligence na Detecção de Pneumonia em Radiografias Torácicas Utilizando DenseNet121 e Grad-CAM

# Detecção de Pneumonia em Radiografias Torácicas com DenseNet121 e Grad-CAM

[cite_start]Este repositório contém a implementação prática e os subsídios metodológicos desenvolvidos para uma dissertação de Mestrado focada na área de **Engenharia Biomédica e Saúde Digital**. [cite_start]O projeto propõe uma abordagem de *Deep Learning* combinada com *Explainable Artificial Intelligence* (XAI) para triagem automatizada e explicável de pneumonia a partir de imagens médicas de raios-X.

[cite_start]O objetivo central é mitigar o problema da "caixa-preta" dos modelos convolucionais tradicionais, utilizando o algoritmo **Grad-CAM** para mapear visualmente as regiões pulmonares que justificaram a tomada de decisão da rede neural.

---

## 📌 Visão Geral do Projeto

* [cite_start]**Arquitetura Base:** DenseNet121 (pré-treinada na ImageNet)[cite: 184, 186].
* [cite_start]**Framework:** TensorFlow / Keras 3[cite: 184].
* [cite_start]**Abordagem de Treinamento:** Transfer Learning seguida de Fine-Tuning das últimas 50 camadas[cite: 186, 188].
* [cite_start]**Explicabilidade:** Grad-CAM (*Gradient-weighted Class Activation Mapping*) mapeando ativações na última camada convolucional relevante[cite: 1, 184, 190].

---

## 📊 Resultados e Gráficos do Modelo

Após a execução do pipeline de treinamento e avaliação, salve os gráficos gerados dentro de uma pasta no repositório (ex: `outputs/` ou `images/`) e substitua as referências abaixo pelos caminhos correspondentes.

### 1. Curvas de Aprendizado (Treinamento vs. Validação)
O comportamento das métricas de perda (*Loss*) e acurácia (*Accuracy*) ao longo das épocas demonstra a estabilidade do aprendizado e o impacto do Fine-Tuning.

```markdown
![Curvas de Acurácia e Perda](outputs/learning_curves.png)

![Visualização Grad-CAM](outputs/gradcam_result.png)

pip install tensorflow matplotlib scikit-learn

DATASET_PATH = r"Caminho/Ate/Seu/Dataset/chest_xray"

