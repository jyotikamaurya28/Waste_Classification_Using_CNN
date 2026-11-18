import os
import random
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Dropout, GlobalAveragePooling2D)

import matplotlib
matplotlib.use("Agg")          # no GUI window, just save images to files
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Dropout, Flatten)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config


def show_samples(train_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for row, cat in enumerate(config.CLASSES):
        folder = os.path.join(train_dir, cat)
        for col, f in enumerate(random.sample(os.listdir(folder), 3)):
            ax = axes[row][col]
            ax.imshow(Image.open(os.path.join(folder, f)))
            ax.set_title(cat)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "samples.png"))
    plt.close()


def show_distribution(train_dir):
    counts = [len(os.listdir(os.path.join(train_dir, c))) for c in config.CLASSES]
    plt.pie(counts, labels=config.CLASSES, autopct="%1.1f%%")
    plt.title("TRAIN Data Class Distribution")
    plt.axis("equal")
    plt.savefig(os.path.join(config.OUTPUT_DIR, "class_distribution.png"))
    plt.close()


def build_model():
    base = MobileNetV2(input_shape=(*config.IMG_SIZE, 3), include_top=False, weights="imagenet")
    for layer in base.layers:
        layer.trainable = False

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(256, kernel_initializer="he_uniform"),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        loss="binary_crossentropy",
        metrics=["accuracy"],
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
    )
    return model


def plot_history(h):
    for metric, title in [("loss", "Model Loss"), ("accuracy", "Model Accuracy")]:
        plt.plot(h.history[metric])
        plt.plot(h.history["val_" + metric])
        plt.title(title)
        plt.ylabel(metric.capitalize())
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.savefig(os.path.join(config.OUTPUT_DIR, f"{metric}.png"))
        plt.close()


def main():
    path = config.get_dataset_path()
    print("Dataset path:", path)
    train_dir = os.path.join(path, "DATASET", "TRAIN")
    test_dir = os.path.join(path, "DATASET", "TEST")

    show_samples(train_dir)
    show_distribution(train_dir)

    train_gen = ImageDataGenerator(rescale=1.0 / 255, zoom_range=0.4, rotation_range=10,
                                   horizontal_flip=True, vertical_flip=True, validation_split=0.2)
    valid_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    common = dict(target_size=config.IMG_SIZE, class_mode="binary", batch_size=config.BATCH_SIZE)
    train_ds = train_gen.flow_from_directory(train_dir, subset="training", **common)
    valid_ds = valid_gen.flow_from_directory(train_dir, subset="validation", **common)
    test_ds = test_gen.flow_from_directory(test_dir, **common)
    print("Class indices:", train_ds.class_indices)   # expect {'O': 0, 'R': 1}

    model = build_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_accuracy", mode="max", patience=2, verbose=1),
        ModelCheckpoint(config.MODEL_PATH, monitor="val_accuracy", mode="max",
                        save_best_only=True, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=valid_ds, epochs=config.EPOCHS,
                        callbacks=callbacks, verbose=1)
    plot_history(history)

    best = load_model(config.MODEL_PATH)
    loss, acc = best.evaluate(test_ds)
    print(f"\nTest loss: {loss:.4f} | Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()