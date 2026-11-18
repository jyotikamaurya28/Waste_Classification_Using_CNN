import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 0.001
CLASSES = ["O", "R"]          # O = Organic (0), R = Recyclable (1)
LOW_CONF = 0.70

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.keras")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_dataset_path():
    """Downloads the dataset from Kaggle (cached after the first run)."""
    import kagglehub
    return kagglehub.dataset_download("techsash/waste-classification-data")