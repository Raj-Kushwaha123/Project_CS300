# Fine-Tuning Strategies for CNN using ResNet18

## 1. Objective

The objective of this experiment is to analyze the impact of different fine-tuning strategies on a pretrained Convolutional Neural Network (CNN). A pretrained ResNet18 model is adapted to the CIFAR10 dataset using three different approaches, and their performance is compared.

---

## 2. Dataset

The experiment was conducted on the **CIFAR10 dataset**, which consists of 60,000 color images of size 32×32 belonging to 10 classes.

| Property        | Value  |
| --------------- | ------ |
| Total Images    | 60,000 |
| Training Images | 50,000 |
| Test Images     | 10,000 |
| Classes         | 10     |
| Image Size      | 32×32  |

The images were resized to 224×224 to match the input requirements of the pretrained model.

---

## 3. Model Architecture

A pretrained **ResNet18** model was used for this experiment.

* The model was pretrained on ImageNet.
* The final fully connected (fc) layer was replaced with a new layer corresponding to 10 output classes.
* Input images were resized to 224×224.

---

## 4. Fine-Tuning Strategies

Three fine-tuning strategies were implemented:

### 4.1 Training Only the Final Classifier (FC Only)

* All pretrained layers were frozen.
* Only the final fully connected layer was trained.
* This approach relies entirely on pretrained features.

---

### 4.2 Unfreezing Last Block + Classifier

* Early layers were frozen.
* The last residual block (layer4) and the classifier were trained.
* Allows adaptation of high-level features.

---

### 4.3 Full Fine-Tuning

* All layers were unfrozen.
* Entire model was trained.
* Provides maximum flexibility for learning.

---

## 5. Training Configuration

| Parameter     | Value            |
| ------------- | ---------------- |
| Model         | ResNet18         |
| Optimizer     | Adam             |
| Learning Rate | 0.001            |
| Loss Function | CrossEntropyLoss |
| Epochs        | 5                |
| Batch Size    | 64               |
| Input Size    | 224×224          |

---

## 6. Experimental Results

### 6.1 Training and Test Accuracy

#### FC Only

| Epoch | Train Accuracy | Test Accuracy |
| ----- | -------------- | ------------- |
| 1     | 72.42%         | 77.66%        |
| 2     | 78.48%         | 79.80%        |
| 3     | 79.54%         | 80.12%        |
| 4     | 79.81%         | 80.64%        |
| 5     | 80.17%         | 80.48%        |

---

#### Last Block + FC

| Epoch | Train Accuracy | Test Accuracy |
| ----- | -------------- | ------------- |
| 1     | 84.67%         | 88.70%        |
| 2     | 92.44%         | 89.01%        |
| 3     | 95.88%         | 88.19%        |
| 4     | 97.25%         | 89.34%        |
| 5     | 97.79%         | 90.46%        |

---

#### Full Fine-Tuning

| Epoch | Train Accuracy | Test Accuracy |
| ----- | -------------- | ------------- |
| 1     | 81.16%         | 85.67%        |
| 2     | 88.93%         | 86.18%        |
| 3     | 92.35%         | 89.80%        |
| 4     | 94.62%         | 87.16%        |
| 5     | 95.63%         | 88.80%        |

---

## 7. Accuracy Comparison

| Strategy         | Best Test Accuracy |
| ---------------- | ------------------ |
| FC Only          | 80.64%             |
| Last Block + FC  | **90.46%**         |
| Full Fine-Tuning | 89.80%             |

---

## 8. Result Analysis

### 8.1 FC Only

* Achieved moderate performance (~80%).
* Limited learning since only the classifier was trained.
* Could not adapt deeper features to the dataset.

---

### 8.2 Last Block + FC

* Achieved the highest accuracy (90.46%).
* Balanced approach between feature reuse and adaptation.
* High-level features were successfully tuned for CIFAR10.

---

### 8.3 Full Fine-Tuning

* Achieved high accuracy but slightly lower than last block.
* Signs of overfitting observed:

  * Training accuracy increased continuously.
  * Test accuracy fluctuated after epoch 3.

---

## 9. Overfitting Analysis

Overfitting is observed in full fine-tuning:

| Epoch | Train Accuracy | Test Accuracy |
| ----- | -------------- | ------------- |
| 3     | 92.35%         | 89.80%        |
| 5     | 95.63%         | 88.80%        |

* Training accuracy increased significantly.
* Test accuracy decreased after peak performance.
* Indicates model memorizing training data.

---

## 10. Key Observations

* Increasing trainable layers improves performance initially.
* Full fine-tuning may lead to overfitting on smaller datasets.
* Partial fine-tuning provides a good trade-off between performance and stability.

---

## 11. Conclusion

This experiment demonstrates the effectiveness of transfer learning and fine-tuning strategies using a pretrained ResNet18 model.

* Training only the classifier yields lower accuracy due to limited adaptability.
* Unfreezing the last block along with the classifier provides the best performance.
* Full fine-tuning achieves high accuracy but introduces overfitting.

Thus, partial fine-tuning is the most effective approach for the CIFAR10 dataset.

---

## 12. Future Work

* Apply data augmentation techniques.
* Use learning rate scheduling.
* Experiment with deeper architectures (ResNet50, EfficientNet).
* Evaluate on larger datasets for better generalization.
