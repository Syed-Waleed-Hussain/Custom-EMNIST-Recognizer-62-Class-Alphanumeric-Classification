# Custom EMNIST Recognizer: 62-Class Alphanumeric Classification 🧠✏️

An end-to-end Machine Learning pipeline that trains a custom Convolutional Neural Network (CNN) from scratch to recognize 62 distinct character classes (digits `0-9`, uppercase `A-Z`, and lowercase `a-z`). 

This project strictly avoids pre-trained weights, demonstrating a fundamental understanding of deep learning architectures, dataset processing, and model evaluation.

## 🚀 Key Features

*   **Custom CNN Architecture:** Built entirely from scratch using TensorFlow/Keras with `Conv2D`, `MaxPooling2D`, `Dropout`, and `Dense` layers.
*   **EMNIST ByClass Dataset:** Handled the extraction, preprocessing, and normalization of ~800,000 images using TensorFlow Datasets (TFDS).
*   **Live Interactive UI:** Integrated a Gradio sketchpad interface directly into the notebook, allowing users to draw characters and get real-time, confidence-scored predictions.
*   **In-Depth Data Analysis:** Includes bar chart visualizations mapping the imbalanced class distribution of the EMNIST ByClass dataset.
*   **Comprehensive Evaluation Matrix:** Features a plotted $62 \times 62$ Confusion Matrix heatmap, alongside a dynamic pandas dataframe calculating True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN) for every single class.

## 🛠️ Tech Stack

*   **Frameworks:** TensorFlow 2.x, Keras
*   **Data Handling:** TensorFlow Datasets (TFDS), NumPy, Pandas
*   **Computer Vision:** OpenCV (`cv2`) for image inversion, resizing, and rotation formatting.
*   **Visualizations:** Matplotlib, Seaborn
*   **Frontend/UI:** Gradio

## 💡 The "EMNIST Quirk" Handled
The EMNIST dataset historically contains images that are rotated 90-degrees clockwise and horizontally flipped. The inference pipeline in this project handles this quirk automatically using OpenCV. When a user draws on the Gradio canvas, the image is mathematically inverted (black-to-white), resized to $28 \times 28$, rotated, and flipped to perfectly match the CNN's expected input dimension `(1, 28, 28, 1)`.

## 📊 Evaluation Highlights
Due to the multi-class nature of this project (62 target variables), standard binary classification metrics are insufficient. The evaluation phase outputs a comprehensive Scikit-Learn `classification_report` detailing **Precision**, **Recall**, and **F1-Score** across all characters, highlighting how the model distinguishes between structurally similar shapes (e.g., `0` vs `O`, `1` vs `l`).

## ⚙️ How to Run
1. Clone this repository.
2. Open the `.ipynb` file in Google Colab or a local Jupyter environment.
3. Ensure hardware acceleration (GPU) is enabled for faster training.
4. Run all cells sequentially. The dataset will download automatically via TFDS.
5. Scroll to the final cell to interact with the Gradio Drawing Canvas.

---
**Author:** Syed Waleed Hussain  
**Domain:** Artificial Intelligence & Data Science
