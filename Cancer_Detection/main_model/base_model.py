import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 0) Enable Apple MPS (TensorFlow + metal plugin sees it as a GPU) ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU(s): {gpus}")
    except RuntimeError as e:
        print("Error setting GPU memory growth:", e)
else:
    print("No GPU detected (falling back to CPU)")

# --- 1) Data‐gathering logic from your second script ---
def get_data_paths_and_labels():
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, 'data')
    csv_dir = os.path.join(data_dir, 'csv_file')
    ds1 = os.path.join(data_dir, 'images_png')               # breast‐level annotations
    ds2 = os.path.join(data_dir, 'train_images_processed_512')  # processed 512px images

    df1 = pd.read_csv(os.path.join(csv_dir, 'breast-level_annotations.csv'))
    df2 = pd.read_csv(os.path.join(csv_dir, 'train.csv'))

    paths, labels = [], []
    # Dataset 1: BIRADS → cancer if ≥ 4
    for _, r in df1.iterrows():
        try:
            lvl = int(str(r['breast_birads']).split()[-1])
        except:
            continue
        lab = 1 if lvl >= 4 else 0
        fn = f"{r['study_id']}_{r['image_id']}.png"
        fp = os.path.join(ds1, fn)
        if os.path.exists(fp):
            paths.append(fp)
            labels.append(lab)

    # Dataset 2: original cancer labels
    for _, r in df2.iterrows():
        fn = f"{int(r['patient_id'])}_{int(r['image_id'])}.png"
        fp = os.path.join(ds2, fn)
        if os.path.exists(fp):
            paths.append(fp)
            labels.append(int(r['cancer']))

    if not paths:
        raise RuntimeError("No images found in either dataset!")
    return paths, np.array(labels, dtype=np.int32)

paths, labs = get_data_paths_and_labels()
df = pd.DataFrame({'image_path': paths, 'cancer': labs})

# --- 2) Stratified train/val/test split ---
train_df, temp_df = train_test_split(
    df, stratify=df['cancer'], test_size=0.30, random_state=42
)
val_df, test_df = train_test_split(
    temp_df, stratify=temp_df['cancer'], test_size=0.50, random_state=42
)

# --- 3) Oversample minority in training set ---
df_major = train_df[train_df.cancer == 0]
df_minor = train_df[train_df.cancer == 1]
df_minor_up = resample(
    df_minor,
    replace=True,
    n_samples=len(df_major),
    random_state=42
)
train_df = pd.concat([df_major, df_minor_up]).sample(frac=1, random_state=42)

# --- 4) Compute class‐weights for loss (optional) ---
y_train = train_df['cancer'].values
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(weights))

# --- 5) ImageDataGenerators & dataframe iterators ---
IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    horizontal_flip=True,
    zoom_range=0.1
)
val_datagen  = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, x_col='image_path', y_col='cancer',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='raw', shuffle=True
)
val_gen = val_datagen.flow_from_dataframe(
    val_df, x_col='image_path', y_col='cancer',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='raw', shuffle=False
)
test_gen = test_datagen.flow_from_dataframe(
    test_df, x_col='image_path', y_col='cancer',
    target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
    class_mode='raw', shuffle=False
)

# --- 6) Build the DenseNet201 model ---
base_model = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

def build_model(backbone, lr=1e-4):
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m = models.Model(inp, out)
    m.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    return m

model = build_model(base_model, lr=1e-4)

# --- 7) Callbacks ---
early_stop = EarlyStopping(
    monitor='val_auc', patience=3, mode='max',
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_auc', factor=0.2, patience=2,
    min_lr=1e-7, mode='max'
)

# --- 8) Train (on MPS/GPU if available) ---
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=4,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# --- Progress update summary ---
print("\n=== Training Progress ===")
for ep in range(len(history.history['loss'])):
    tr_loss = history.history['loss'][ep]
    tr_acc  = history.history['accuracy'][ep]
    tr_auc  = history.history['auc'][ep]
    vl_loss = history.history['val_loss'][ep]
    vl_acc  = history.history['val_accuracy'][ep]
    vl_auc  = history.history['val_auc'][ep]
    print(f"Epoch {ep+1:>2}  "
          f"Train → loss: {tr_loss:.4f}, acc: {tr_acc:.4f}, auc: {tr_auc:.4f} |  "
          f"Val   → loss: {vl_loss:.4f}, acc: {vl_acc:.4f}, auc: {vl_auc:.4f}")

# --- 9) Evaluate on test set & get predictions ---
results = model.evaluate(test_gen, verbose=0)
print(f"\nTest Loss: {results[0]:.4f}  "
      f"Test Accuracy: {results[1]:.4f}  "
      f"Test AUC: {results[2]:.4f}")

probs  = model.predict(test_gen).ravel()
y_true = test_df['cancer'].values
y_pred = (probs > 0.5).astype(int)

# --- 10) Classification report ---
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# --- 11) Confusion matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['No Cancer','Cancer'],
    yticklabels=['No Cancer','Cancer']
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# --- 12) F1 score ---
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")

# --- 13) ROC curve & AUC ---
fpr, tpr, _ = roc_curve(y_true, probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
