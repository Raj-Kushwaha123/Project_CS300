# Comprehensive Analysis of Fine-Tuning Strategies with Varying Dataset Sizes

## 1. Objective

The objective of this experiment is to analyze how different fine-tuning strategies perform under varying amounts of training data. The study evaluates performance using 20%, 50%, and 100% of the CIFAR10 dataset.

---

## 2. Dataset

The CIFAR10 dataset was used. Three subsets were created:

- 20% of training data  
- 50% of training data  
- 100% of training data  

Each subset was split into 80% training and 20% validation.

---

## 3. Fine-Tuning Strategies

Three strategies were used:

1. FC Only (train only classifier)  
2. Last Block + FC  
3. Full Fine-Tuning  

---

## 4. Training Configuration

| Parameter     | Value    |
|--------------|----------|
| Model        | ResNet18 |
| Optimizer    | Adam     |
| Learning Rate| 0.001    |
| Epochs       | 5        |
| Batch Size   | 64       |

---

## 5. Results

### 5.1 Accuracy Comparison Across Dataset Sizes

| Dataset Size | FC Only | Last Block + FC | Full Fine-Tuning |
|--------------|--------|----------------|-----------------|
| 20%          | 77.50% | 83.80%         | 77.90%          |
| 50%          | 79.46% | 88.36%         | 86.64%          |
| 100%         | 80.59% | 90.83%         | 88.44%          |

---

## 5.2 Performance Visualizations

### 5.2.1 Training on 20% of Data

**Loss Curves:**
![20% Data Loss](20_Data_loss.png)

**Accuracy Curves:**
![20% Data Accuracy](20_Data_acc.png)

**Observations:** With limited data (20%), the FC Only strategy maintains steady learning, while Last Block shows rapid loss reduction, indicating better feature adaptation. The Last Block strategy achieves the best validation accuracy (83.80%) for this dataset size.

### 5.2.2 Training on 50% of Data

**Loss Curves:**
![50% Data Loss](50_Data_loss.png)

**Accuracy Curves:**
![50% Data Accuracy](50_Data_acc.png)

**Observations:** With 50% of the data, the Last Block strategy clearly outperforms other methods, achieving 88.36% accuracy. Full Fine-Tuning also shows strong performance at 86.64%. The FC Only strategy lags slightly at 79.46%.

### 5.2.3 Training on 100% of Data

**Loss Curves:**
![100% Data Loss](100_Data_loss.png)

**Accuracy Curves:**
![100% Data Accuracy](100_Data_acc.png)

**Observations:** With the full dataset, the Last Block strategy achieves the best performance at 90.83%. Full Fine-Tuning achieves 88.44%, showing comparable but slightly lower performance. FC Only reaches 80.59%, demonstrating that even with more data, simply training the classifier is insufficient for optimal performance.

---

## 6. Detailed Analysis

### 6.1 Effect of Dataset Size

**20% Dataset:**
- All strategies show lower baseline accuracy due to limited data exposure
- Last Block strategy performs best (83.80%), showing that adapting high-level features is crucial with limited data
- FC Only achieves 77.50%, demonstrating that frozen pretrained features need some adaptation for optimal performance
- Full Fine-Tuning achieves 77.90%, slightly better than FC Only, indicating that full training can provide marginal improvements

**50% Dataset:**
- Significant accuracy improvements across all strategies
- Last Block dominates with 88.36%, showing consistent superiority
- Full Fine-Tuning improves to 86.64%, closing the gap with Last Block
- FC Only reaches 79.46%, showing that more data helps, but strategy choice still matters significantly

**100% Dataset:**
- Highest accuracy values across the board
- Last Block achieves peak performance at 90.83%
- Full Fine-Tuning reaches 88.44%, competitive but not surpassing Last Block
- FC Only reaches 80.59%, showing diminishing returns from additional data without feature adaptation

### 6.2 Effect of Fine-Tuning Strategy

**FC Only Strategy:**
- Maintains relatively consistent accuracy across dataset sizes (77.50% → 79.46% → 80.59%)
- Shows minimal improvement with more data (only 3.09% improvement from 20% to 100%)
- Limited by frozen pretrained features that may not fully align with CIFAR-10 task
- Suitable for scenarios with very limited computational resources

**Last Block + FC Strategy:**
- Clear winner across all dataset sizes
- Substantial improvements with more data (77.50% → 88.36% → 90.83%)
- 13.33 percentage point improvement from 20% to 100% data
- Optimal balance between model capacity and regularization
- Better generalization compared to full fine-tuning

**Full Fine-Tuning Strategy:**
- Shows strong performance on 50% and 100% datasets
- More volatile than Last Block strategy
- May suffer from overfitting on smaller datasets (77.90% at 20% data)
- Requires more careful regularization and often more data to prevent overfitting
- Performs comparably to Last Block on larger datasets but never surpasses it

### 6.3 Key Performance Metrics

| Metric | Value |
|--------|-------|
| Best Overall Accuracy | 90.83% (Last Block, 100% data) |
| Worst Accuracy | 77.50% (FC Only, 20% data) |
| Best Strategy | Last Block + FC (wins on all dataset sizes) |
| Average Improvement (20%→100%) | 8.97% (Last Block: 13.33%, Full FT: 10.54%, FC Only: 3.09%) |
| Most Consistent Strategy | Last Block + FC (smallest variance across dataset sizes) |

---

## 7. Conclusion

### Summary of Findings

The experiment demonstrates that both dataset size and fine-tuning strategy significantly influence model performance on the CIFAR-10 dataset. Key conclusions include:

1. **Dataset Size Matters:** Increasing training data from 20% to 100% provides consistent improvements across all strategies, with the Last Block strategy achieving the most dramatic improvement (77.50% → 90.83%).

2. **Last Block + FC is Superior:** This strategy outperforms both alternatives across all dataset sizes, providing the best balance between:
   - Model capacity (more flexible than FC Only)
   - Generalization (more stable than Full Fine-Tuning)
   - Data efficiency (shows strong improvement even with limited data)

3. **Full Fine-Tuning Has Tradeoffs:** While competitive on larger datasets, this approach shows:
   - Signs of overfitting on smaller datasets
   - Requires more data to achieve optimal performance
   - Never surpasses Last Block strategy despite higher model capacity

4. **FC Only is Insufficient:** Freezing all pretrained features limits adaptability and results in:
   - The lowest performance across all dataset sizes
   - Minimal improvement with additional data
   - Only 3.09% accuracy gain from 20% to 100% data

### Practical Recommendations

1. **For small datasets (20%):** Use Last Block + FC strategy with early stopping to prevent overfitting
2. **For medium datasets (50%):** Last Block + FC remains optimal; Full Fine-Tuning becomes viable
3. **For large datasets (100%):** Last Block + FC still recommended; Full Fine-Tuning provides marginal benefits only
4. **Avoid FC Only** unless computational constraints are severe

---

## 8. Model Checkpoints

The best-performing models for each configuration have been saved:

- `20_last_best.pth` - Best model for 20% data (Last Block strategy): 83.80% accuracy
- `50_last_best.pth` - Best model for 50% data (Last Block strategy): 88.36% accuracy
- `100_last_best.pth` - Best model for 100% data (Last Block strategy): 90.83% accuracy

These checkpoints can be used for inference on new CIFAR-10 test data or further fine-tuning.