# ♻️ Waste Classification using CNN

A deep learning project that classifies waste images as **Organic** or **Recyclable** using transfer learning (MobileNetV2), with an interactive **Gradio** web app for real-time predictions and disposal tips.

Live Link- https://waste-classification-cnn-478z.onrender.com/
## 📌 Features
- Binary image classification: Organic (O) vs Recyclable (R)
- Transfer learning with MobileNetV2 (ImageNet weights)
- Data augmentation (zoom, rotation, flips) for better generalization
- Early stopping and best-model checkpointing
- Gradio web UI: upload, webcam or paste an image
- Confidence score, low-confidence warning, disposal tips, session stats and scan history


## 📊 Dataset
[Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data) from Kaggle. It has two classes, Organic (`O`) and Recyclable (`R`), split into TRAIN and TEST folders. The dataset downloads automatically through `kagglehub` when you run `train.py`.

## 🧠 Model
| Item | Details |

| Base model | MobileNetV2 (ImageNet, frozen) |
| Head | GlobalAveragePooling → BatchNorm → Dense(256) → ReLU → Dropout(0.3) → Dense(1, sigmoid) |
| Loss | Binary cross-entropy |
| Optimizer | Adam (lr = 0.001) |
| Input size | 224 × 224 |
| Validation accuracy | **94.39%** |
| Test accuracy | **86.87%** |

## 📁 Project Structure
```
Waste_Classification_Using_CNN/
├── app.py              # Gradio web app
├── train.py            # Model training and evaluation
├── predict.py          # Single-image prediction (CLI)
├── config.py           # Settings and dataset download
├── best_model.keras    # Trained model
├── requirements.txt
├── outputs/            # Training plots
└── assets/             # README images
```

## ⚙️ Installation
Requires **Python 3.10 – 3.12**.

```bash
git clone https://github.com/jyotikamaurya28/Waste_Classification_Using_CNN.git
cd Waste_Classification_Using_CNN

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

python -m pip install -r requirements.txt
```

## 🚀 Usage
**Train the model** (optional, a trained model is included):
```bash
python train.py
```

**Predict a single image:**
```bash
python predict.py path/to/image.jpg
```

**Run the web app:**
```bash
python app.py
```
Then open http://127.0.0.1:7860

## 🛠️ Tech Stack
Python · TensorFlow / Keras · Gradio · NumPy · Matplotlib · Pillow

## 🔮 Future Improvements
- Multi-class classification (plastic, paper, glass, metal, etc.)
- Deploy on Hugging Face Spaces
- Fine-tune the base model layers for higher accuracy

## 👩‍💻 Author
**Jyotika Maurya**
GitHub: [@jyotikamaurya28](https://github.com/jyotikamaurya28)

## 📄 License
This project is licensed under the MIT License.
