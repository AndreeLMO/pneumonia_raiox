# ============================================================
# DETECÇÃO DE PNEUMONIA + GRAD-CAM
# DenseNet121 + TensorFlow + Matplotlib
# KERAS 3 COMPATÍVEL
# ============================================================

# ============================================================
# INSTALAÇÃO (caso necessário)
# ============================================================

# pip install tensorflow matplotlib scikit-learn

# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from tensorflow.keras.models import (
    Model,
    load_model
)

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.applications import DenseNet121

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator
)

from tensorflow.keras.preprocessing import image

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ============================================================
# CONFIGURAÇÕES
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

# ============================================================
# CAMINHO DO DATASET
# ============================================================

DATASET_PATH = r"D:\archive\chest_xray"

TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# ============================================================
# PASTA PARA SALVAR MODELO
# ============================================================

SAVE_DIR = r"D:\Projetos\Modelos"

os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    SAVE_DIR,
    "modelo_pneumonia_densenet121.keras"
)

# ============================================================
# DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ============================================================
# GERADORES
# ============================================================

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ============================================================
# CLASSES
# ============================================================

print("\nClasses:")
print(train_generator.class_indices)

# ============================================================
# DENSENET121
# ============================================================

base_model = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Congelar camadas iniciais
base_model.trainable = False

# ============================================================
# CAMADAS FINAIS
# ============================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

x = Dense(128, activation='relu')(x)

x = Dropout(0.3)(x)

output = Dense(1, activation='sigmoid')(x)

# ============================================================
# MODELO FINAL
# ============================================================

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ============================================================
# COMPILAÇÃO
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# ============================================================
# RESUMO
# ============================================================

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor='val_auc',
    mode='max',
    save_best_only=True
)

callbacks = [
    early_stop,
    reduce_lr,
    checkpoint
]

# ============================================================
# TREINAMENTO
# ============================================================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# FINE-TUNING
# ============================================================

print("\nIniciando Fine-Tuning...\n")

base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=callbacks
)

# ============================================================
# SALVAR MODELO FINAL
# ============================================================

model.save(MODEL_PATH)

print("\nModelo salvo com sucesso!")
print(MODEL_PATH)

# ============================================================
# CARREGAR MODELO
# ============================================================

model = load_model(MODEL_PATH)

print("\nModelo carregado com sucesso!")

# ============================================================
# AVALIAÇÃO
# ============================================================

results = model.evaluate(test_generator)

print("\n========== RESULTADOS ==========")

for metric, value in zip(model.metrics_names, results):
    print(f"{metric}: {value:.4f}")

# ============================================================
# PREDIÇÕES
# ============================================================

pred_probs = model.predict(test_generator)

predictions = (pred_probs > 0.5).astype(int)

y_true = test_generator.classes

# ============================================================
# RELATÓRIO
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(classification_report(
    y_true,
    predictions,
    target_names=['NORMAL', 'PNEUMONIA']
))

# ============================================================
# ROC-AUC
# ============================================================

auc = roc_auc_score(y_true, pred_probs)

print(f"\nROC-AUC: {auc:.4f}")

# ============================================================
# CURVAS
# ============================================================

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')

plt.title('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')

plt.title('Loss')
plt.legend()

plt.show()

# ============================================================
# GRAD-CAM
# ============================================================

print("\n==============================")
print("INICIANDO GRAD-CAM")
print("==============================")

# ============================================================
# MOSTRAR ÚLTIMAS CAMADAS
# ============================================================

print("\nÚltimas camadas do modelo:\n")

for layer in model.layers[-20:]:
    print(layer.name)

# ============================================================
# ÚLTIMA CAMADA CONVOLUCIONAL
# ============================================================

LAST_CONV_LAYER = "relu"

# ============================================================
# PRÉ-PROCESSAMENTO
# ============================================================

def preprocess_image(img_path):

    img = image.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img, img_array

# ============================================================
# HEATMAP
# ============================================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name
):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        class_idx = tf.argmax(predictions[0])

        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# ============================================================
# VISUALIZAÇÃO
# ============================================================

def visualizar_gradcam(img_path):

    original_img, img_array = preprocess_image(img_path)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        classe = "PNEUMONIA"
        confianca = prediction * 100
    else:
        classe = "NORMAL"
        confianca = (1 - prediction) * 100

    print("\n===================================")
    print(f"Classe prevista: {classe}")
    print(f"Confiança: {confianca:.2f}%")
    print("===================================\n")

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        LAST_CONV_LAYER
    )

    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (IMG_SIZE, IMG_SIZE)
    )

    heatmap = tf.squeeze(heatmap)

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(figsize=(15,5))

    # Original
    plt.subplot(1,3,1)

    plt.imshow(original_img)

    plt.title("Imagem Original")

    plt.axis("off")

    # Heatmap
    plt.subplot(1,3,2)

    plt.imshow(heatmap, cmap='jet')

    plt.title("Heatmap")

    plt.axis("off")

    # Grad-CAM
    plt.subplot(1,3,3)

    plt.imshow(original_img)

    plt.imshow(
        heatmap,
        cmap='jet',
        alpha=0.4
    )

    plt.title(f"Grad-CAM\n{classe}")

    plt.axis("off")

    plt.show()

# ============================================================
# TESTAR GRAD-CAM
# ============================================================

IMG_PATH = r"D:\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg"

visualizar_gradcam(IMG_PATH)

# ============================================================
# FIM
# ============================================================