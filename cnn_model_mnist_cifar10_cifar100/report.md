# CNN Training and Evaluation on MNIST, CIFAR10, and CIFAR100

## 1. Objective

The objective of this assignment is to implement and train **Convolutional Neural Network (CNN)** models on three standard image classification datasets: **MNIST**, **CIFAR10**, and **CIFAR100**.

The tasks performed in this assignment include:

1. Training CNN models separately on each dataset.
2. Saving the **best model checkpoints** based on validation performance.
3. Evaluating the trained models on the **test datasets**.
4. Recording and analyzing the performance results.

These datasets are commonly used benchmarks to evaluate the performance of deep learning models for image classification tasks.

---

# 2. Datasets Used

## 2.1 MNIST Dataset

The MNIST dataset contains grayscale images of handwritten digits from 0 to 9. It is widely used as a beginner dataset for image classification tasks.

**Dataset Details**

| Property          | Value         |
| ----------------- | ------------- |
| Total Images      | 70,000        |
| Training Images   | 60,000        |
| Testing Images    | 10,000        |
| Image Size        | 28 × 28       |
| Channels          | 1 (Grayscale) |
| Number of Classes | 10            |

---

## 2.2 CIFAR10 Dataset

The CIFAR10 dataset contains colored images representing various real-world objects such as airplanes, cars, birds, cats, dogs, and trucks.

**Dataset Details**

| Property          | Value   |
| ----------------- | ------- |
| Total Images      | 60,000  |
| Training Images   | 50,000  |
| Testing Images    | 10,000  |
| Image Size        | 32 × 32 |
| Channels          | 3 (RGB) |
| Number of Classes | 10      |

---

## 2.3 CIFAR100 Dataset

The CIFAR100 dataset is similar to CIFAR10 but contains **100 different object categories**, making it significantly more challenging.

**Dataset Details**

| Property          | Value   |
| ----------------- | ------- |
| Total Images      | 60,000  |
| Training Images   | 50,000  |
| Testing Images    | 10,000  |
| Image Size        | 32 × 32 |
| Channels          | 3 (RGB) |
| Number of Classes | 100     |

---

# 3. Model Architecture

A **Convolutional Neural Network (CNN)** was implemented for image classification.

The architecture consists of the following components:

1. **Convolutional Layers**

   * Extract spatial features from images.
   * Detect patterns such as edges, textures, and shapes.

2. **ReLU Activation Function**

   * Introduces non-linearity to the network.

3. **Max Pooling Layers**

   * Reduce spatial dimensions of feature maps.
   * Help reduce computation and overfitting.

4. **Fully Connected Layers**

   * Perform classification based on extracted features.

**Architecture Flow**

Input Image
→ Convolution Layer (32 filters) + ReLU
→ Max Pooling
→ Convolution Layer (64 filters) + ReLU
→ Max Pooling
→ Convolution Layer (128 filters) + ReLU
→ Max Pooling
→ Flatten
→ Fully Connected Layer (256 neurons)
→ Output Layer (Classification)

---

# 4. Data Preprocessing

Different preprocessing techniques were applied for the datasets.

### MNIST

* Images were resized to **32 × 32** to match the CNN input structure.
* Images were converted to **tensor format**.

### CIFAR10 and CIFAR100

* Images were directly converted to **tensor format**.

---

# 5. Dataset Splitting

Each dataset was divided into:

| Split          | Percentage |
| -------------- | ---------- |
| Training Set   | 80%        |
| Validation Set | 20%        |

The validation set was used to monitor model performance during training.

---

# 6. Training Configuration

The CNN models were trained using the following hyperparameters:

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Optimizer        | Adam                     |
| Learning Rate    | 0.001                    |
| Loss Function    | CrossEntropyLoss         |
| Batch Size       | 64                       |
| Number of Epochs | 10                       |
| Hardware         | GPU (if available) / CPU |

---

# 7. Model Checkpointing

During training, the model was evaluated on the **validation dataset after every epoch**.

If the validation accuracy improved compared to the previous best value, the model weights were saved as a checkpoint.

Example checkpoint files:

```
MNIST_best.pth
CIFAR10_best.pth
CIFAR100_best.pth
```

This ensures that the best performing model is preserved.

---

# 8. Model Evaluation

After training, the saved best models were loaded and evaluated on the **test datasets**.

The evaluation metric used was **classification accuracy**.

Accuracy is calculated as:

Accuracy = (Correct Predictions) / (Total Predictions)

---

# 9. Experimental Results

The trained models produced the following test accuracies:

| Dataset  | Test Accuracy | Percentage |
| -------- | ------------- | ---------- |
| MNIST    | 0.9917        | 99.17%     |
| CIFAR10  | 0.7173        | 71.73%     |
| CIFAR100 | 0.3825        | 38.25%     |

---

# 10. Result Analysis

### MNIST

The model achieved **99.17% accuracy**, which is expected since MNIST is a relatively simple dataset with grayscale images and limited variations.

### CIFAR10

The model achieved **71.73% accuracy**. CIFAR10 contains more complex objects and colored images, making the classification task more challenging.

### CIFAR100

The model achieved **38.25% accuracy**. This dataset is significantly more difficult because it contains **100 classes**, leading to fewer training examples per class and greater classification complexity.

---

# 11. Challenges

Several challenges were encountered during the training process:

* The simple CNN architecture struggled with more complex datasets.
* CIFAR datasets contain low-resolution images which make feature extraction harder.
* CIFAR100 has many categories, increasing classification difficulty.

---

# 12. Possible Improvements

Model performance could be improved by:

* Using deeper architectures such as **ResNet or VGG**.
* Applying **data augmentation techniques**.
* Increasing the number of training epochs.
* Using **learning rate scheduling**.
* Applying **regularization techniques** to reduce overfitting.

---

# 13. Conclusion

In this assignment, CNN models were successfully trained and evaluated on three standard datasets: **MNIST, CIFAR10, and CIFAR100**.

The results demonstrate that CNN models perform very well on simpler datasets such as MNIST, while performance decreases as dataset complexity increases. More advanced architectures and training strategies are required to achieve better performance on challenging datasets such as CIFAR100.

This experiment highlights the importance of dataset complexity, model design, and training strategies in deep learning-based image classification tasks.
