# ============================================================
# DETECÇÃO DE PNEUMONIA + GRAD-CAM
# DenseNet121 + TensorFlow + Matplotlib
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ============================================================
# CONFIGURAÇÕES
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
DATASET_PATH = r"D:\archive\chest_xray"
SAVE_DIR = r"D:\Projetos\Modelos"
MODEL_PATH = os.path.join(SAVE_DIR, "modelo_pneumonia_densenet121.keras")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# DATA AUGMENTATION & GERADORES
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print("\nClasses:", train_generator.class_indices)

# ============================================================
# MODELO DENSENET121
# ============================================================

def build_model(trainable=False, lr=1e-4):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = trainable

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

model = build_model(trainable=False)

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    ModelCheckpoint(filepath=MODEL_PATH, monitor='val_auc', mode='max', save_best_only=True)
]

# ============================================================
# TREINAMENTO + FINE-TUNING
# ============================================================

history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=callbacks)

print("\nIniciando Fine-Tuning...\n")
model = build_model(trainable=True, lr=1e-5)
for layer in model.layers[:-50]:
    layer.trainable = False

history_fine = model.fit(train_generator, validation_data=val_generator, epochs=5, callbacks=callbacks)

model.save(MODEL_PATH)
print("\nModelo salvo com sucesso!")

# ============================================================
# AVALIAÇÃO
# ============================================================

model = load_model(MODEL_PATH)
results = model.evaluate(test_generator)
print("\n========== RESULTADOS ==========")
for metric, value in zip(model.metrics_names, results):
    print(f"{metric}: {value:.4f}")

pred_probs = model.predict(test_generator)
predictions = (pred_probs > 0.5).astype(int)
y_true = test_generator.classes

print("\n========== CLASSIFICATION REPORT ==========\n")
print(classification_report(y_true, predictions, target_names=['NORMAL', 'PNEUMONIA']))
print(f"\nROC-AUC: {roc_auc_score(y_true, pred_probs):.4f}")

# ============================================================
# CURVAS DE TREINAMENTO
# ============================================================

def plot_history(history, title="Treinamento"):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{title} - Loss')
    plt.legend()
    plt.show()

plot_history(history, "Inicial")
plot_history(history_fine, "Fine-Tuning")

# ============================================================
# GRAD-CAM
# ============================================================

LAST_CONV_LAYER = "relu"

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    return img, np.expand_dims(img_array, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(inputs=model.inputs,
                       outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.squeeze(conv_outputs @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def visualizar_gradcam(img_path):
    original_img, img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)[0][0]
    classe = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confianca = prediction*100 if prediction > 0.5 else (1-prediction)*100
    print(f"\nClasse prevista: {classe} | Confiança: {confianca:.2f}%")

    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
    heatmap = tf.image.resize(heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE))
    heatmap = tf.squeeze(heatmap)

    plt.figure(figsize=(15,5))
    for i, (img, title) in enumerate([(original_img, "Imagem Original"),
                                      (heatmap, "Heatmap"),
                                      (original_img, f"Grad-CAM\n{classe}")]):
        plt.subplot(1,3,i+1)
        plt.imshow(img if i!=2 else original_img)
        if i==2: plt.imshow(heatmap, cmap='jet', alpha=0.4)
        else: plt.imshow(img, cmap='jet' if i==1 else None)
        plt.title(title)
        plt.axis("off")
    plt.show()

# ============================================================
# TESTAR GRAD-CAM
# ============================================================

IMG_PATH = r"D:\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg"
visualizar_gradcam(IMG_PATH)