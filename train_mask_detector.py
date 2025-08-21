import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---------- SETTINGS ----------
DATASET_DIR = r"C:\Users\91812\OneDrive\Desktop\face-mask-detector\dataset"  # <<< adjust if needed
MODEL_OUT_DIR = r"C:\Users\91812\OneDrive\Desktop\face-mask-detector\models"
MODEL_OUT_PATH = os.path.join(MODEL_OUT_DIR, "mask_detector.h5")

IMG_SIZE = 150   # keep 150x150 to match your current inference code
BATCH_SIZE = 32
VAL_SPLIT = 0.20
EPOCHS_STAGE1 = 10      # train top layers
EPOCHS_STAGE2 = 5       # fine-tune last layers of backbone
LEARNING_RATE_1 = 1e-4
LEARNING_RATE_2 = 1e-5
SEED = 42
# ------------------------------

os.makedirs(MODEL_OUT_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# 1) DATA: augmentation + split
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    shear_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=VAL_SPLIT
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=SEED
)

val_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=SEED
)

print("\nClass mapping:", train_gen.class_indices)  # e.g. {'with_mask': 0, 'without_mask': 1}

# Optional class weighting (handles imbalance)
counts = np.bincount(train_gen.classes)
total = counts.sum()
class_weight = {i: total / (2.0 * counts[i]) for i in range(len(counts))}
print("Class counts:", counts, " -> class_weight:", class_weight)

# 2) MODEL: MobileNetV2 backbone
base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3))
)
for layer in base.layers:
    layer.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
out = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base.input, outputs=out)
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_1),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# Callbacks
ckpt = ModelCheckpoint(
    MODEL_OUT_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)
early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6, verbose=1)

# 3) TRAIN (stage 1: head only)
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weight,
    callbacks=[ckpt, early, reduce],
    verbose=1
)

# 4) FINE-TUNE (stage 2: unfreeze last N layers of backbone)
# Unfreeze last 20 layers (except BatchNorm layers for stability)
unfreeze_from = max(0, len(base.layers) - 20)
for i, layer in enumerate(base.layers):
    if i >= unfreeze_from and not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_2),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weight,
    callbacks=[ckpt, early, reduce],
    verbose=1
)

# 5) FINAL EVAL + SAVE
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"\nValidation accuracy: {val_acc:.4f}  |  loss: {val_loss:.4f}")

# Ensure best model is saved (ModelCheckpoint already saved best)
model.save(MODEL_OUT_PATH)
print(f"\n✅ Saved model to: {MODEL_OUT_PATH}")
