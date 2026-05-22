# Aplicação de Deep Learning e Explainable Artificial Intelligence na Detecção de Pneumonia em Radiografias Torácicas Utilizando DenseNet121 e Grad-CAM

Este repositório apresenta o desenvolvimento de um sistema inteligente para detecção automática de pneumonia em radiografias torácicas, utilizando Deep Learning com a arquitetura DenseNet121 e técnicas de Explainable Artificial Intelligence (XAI), em especial o Grad-CAM para interpretabilidade visual.

📌 Motivação
A pneumonia é uma das principais causas de mortalidade por doenças respiratórias em escala global, especialmente em crianças, idosos e imunocomprometidos. O diagnóstico precoce é essencial, mas a interpretação de radiografias pode ser complexa e sujeita à variabilidade entre especialistas.

Com o avanço da Inteligência Artificial (IA), tornou-se possível criar sistemas capazes de identificar padrões complexos em imagens médicas, auxiliando profissionais da saúde e aumentando a confiabilidade diagnóstica.

⚙️ Tecnologias Utilizadas
Python

TensorFlow / Keras

DenseNet121 (Transfer Learning)

Grad-CAM (Explainable AI)

OpenCV / Matplotlib para visualização

Chest X-Ray Pneumonia Dataset como base de dados

📂 Estrutura do Projeto
data/ → Dataset de radiografias (NORMAL e PNEUMONIA)

notebooks/ → Jupyter Notebooks com experimentos e treinamento

models/ → Arquitetura DenseNet121 e pesos treinados

gradcam/ → Implementação da técnica Grad-CAM para interpretabilidade

results/ → Métricas, matrizes de confusão e exemplos de mapas de ativação

🧪 Metodologia
Pré-processamento das imagens

Redimensionamento para 224×224 pixels

Normalização dos valores de pixel

Data Augmentation (rotações, zoom, deslocamentos, espelhamento)

Arquitetura do Modelo

DenseNet121 pré-treinada no ImageNet

Camadas iniciais congeladas

Adição de camadas totalmente conectadas (GAP, Dense, Dropout, Sigmoid)

Fine-Tuning das últimas camadas para especialização médica

Avaliação

Métricas: Accuracy, Precision, Recall, F1-score, ROC-AUC

Matriz de confusão para análise detalhada

Explainable AI (Grad-CAM)

Geração de mapas de ativação sobrepostos às radiografias

Identificação das regiões pulmonares relevantes para a decisão da rede

📊 Resultados
Alta acurácia e sensibilidade na detecção de pneumonia

Fine-Tuning melhorou significativamente o desempenho em relação ao treinamento apenas das camadas finais

Grad-CAM aumentou a interpretabilidade, mostrando que a rede concentrou atenção em regiões pulmonares compatíveis com opacidades e inflamações

🚀 Futuro do Projeto
Utilização de bases de dados maiores e multicêntricas

Aplicação de segmentação pulmonar para maior precisão

Comparação com arquiteturas modernas (EfficientNet, Vision Transformers)

Desenvolvimento de aplicação web para suporte clínico em tempo real

Integração com sistemas hospitalares e prontuários eletrônicos

📖 Conclusão
Este projeto demonstra o potencial da IA aplicada à saúde digital, combinando CNNs profundas com XAI para oferecer não apenas alta performance, mas também transparência e confiabilidade clínica.

O sistema pode servir como ferramenta de apoio ao diagnóstico médico, contribuindo para maior precisão e suporte à decisão em ambientes hospitalares.

👨‍💻 Autor
Projeto desenvolvido por André.
Este repositório é parte de estudos e experimentos em Deep Learning aplicado à radiologia computacional.
