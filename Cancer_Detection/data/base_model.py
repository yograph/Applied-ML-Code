import pandas as pd
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.applications import DenseNet201
from tf_keras.models import Sequential
from tf_keras import layers, models
from tf_keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tf_keras.metrics import AUC
from sklearn.utils import resample
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


## load the data
TRAIN_PATH = "/kaggle/input/rsna-breast-cancer-detection/train.csv"
train_data = pd.read_csv(TRAIN_PATH)
# print(train_data.head())

## check and handle class imbalance
cancer_group = train_data["cancer"] == 1
non_cancer_group = train_data["cancer"] == 0
# print(cancer_group.sum())
# print(non_cancer_group.sum())
labels = train_data["cancer"].values
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(labels), y=labels
)
# print(class_weights) # [ 0.51081273 23.6208981 ]
class_weights_dict = dict(enumerate(class_weights))

## add new col: image file
IMAGE_PATH = "/kaggle/input/rsna-breast-cancer-512-pngs"
train_data["image_file"] = train_data.apply(
    lambda row: f"{row['patient_id']}_{row['image_id']}.png", axis=1
)
# print(train_data.head())
train_data["image_path"] = train_data["image_file"].apply(
    lambda image_file: os.path.join(IMAGE_PATH, image_file)
)
# print(train_data.head())
# if os.path.exists(train_data["image_path"][0]):
#     print("it exists!!")

## split the data
train_df, tem_df = train_test_split(
    train_data, stratify=train_data["cancer"], test_size=0.3, random_state=42
)
val_df, test_df = train_test_split(
    tem_df, stratify=tem_df["cancer"], test_size=0.5, random_state=42
)

# Separate majority and minority classes
df_majority = train_df[train_df.cancer == 0]
df_minority = train_df[train_df.cancer == 1]
df_minority_resampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)
train_df  = pd.concat([df_majority, df_minority_resampled])
train_df  = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

# print(f"train_df: \n {train_df.head()}")
# print(f"val_df: \n {val_df.head()}")
# print(f"test_df: \n {test_df.head()}")

IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, rotation_range=10, horizontal_flip=True, zoom_range=0.1
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="image_path",
    y_col="cancer",
    directory=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw",
    # shuffle=True  <-- good for training!
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="image_path",
    y_col="cancer",
    directory=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False,
)


test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="image_path",
    y_col="cancer",
    directory=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False,
)

basemodel = DenseNet201(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

basemodel.trainable = False

def build_model(backbone, lr=1e-4):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False) # Make sure it's not training during freeze phase
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.BatchNormalization()(x)
    outputs =  layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", AUC(name='auc')]
    )
    return model


model = build_model(basemodel, lr=1e-4)
# model.summary()


#--------------------------no weights---------------------------
# history_no_weights = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=1
# )

# result = model.evaluate(test_generator)
# print(f"evaluation results no_weights: {result}")


# y_pred_test = model.predict(test_generator)
# y_pred_test = (y_pred_test > 0.5).astype(int)
# print("Classification Report no_weights (Test Set):")
# print(classification_report(test_df['cancer'], y_pred_test))


# cm_test = confusion_matrix(test_df['cancer'], y_pred_test)
# sns.heatmap(cm_test, annot=True, fmt="d", cmap='Blues', xticklabels=["No Cancer", "Cancer"], yticklabels=["No Cancer", "Cancer"])
# plt.title('Confusion Matrix no_weights (Test Set)')
# plt.show()

#--------------------------train model---------------------------

# early_stop = EarlyStopping(
#     monitor='val_auc',       # or 'val_loss' if AUC isn't improving
#     patience=3,
#     restore_best_weights=True,
#     mode='max'               # because higher AUC is better
# )

# reduce_lr = ReduceLROnPlateau(
#     monitor='val_auc',
#     factor=0.2,
#     patience=2,
#     min_lr=1e-7,
#     mode='max'
# )

history_with_weights = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)


# result = model.evaluate(test_generator)
# print(f"evaluation results with_weights: {result}")


y_pred_test = model.predict(test_generator)
y_pred_test = (y_pred_test > 0.5).astype(int)
print("Classification Report with_weights (Test Set):")
print(classification_report(test_df['cancer'], y_pred_test))