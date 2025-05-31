### Evaluation & Visualization

After training our Random Forest classifier with SMOTE‐resampled data and an adjusted probability threshold (0.27), we evaluate performance on the untouched 20 % test set to simulate real‐world conditions (i.e., the original 96.71 % non‐bankrupt vs. 3.29 % bankrupt class imbalance). The following metrics and plots help us understand how well the model identifies bankrupt companies while minimizing false alarms.

**Confusion Matrix Interpretation**  
The confusion matrix in `models/rf_results/confusion_matrix.png` shows:
- **True Positives (TP):** Out of 42 actual bankrupt cases, the model correctly flags 34 → **81 % recall** on the minority class.  
- **True Negatives (TN):** Out of 1,291 non‐bankrupt companies, 1,204 are correctly classified → **93 % recall** on the majority class.  
  
  This tells us that, by lowering the decision threshold to 0.27, we successfully increased recall for bankrupt firms (reducing the cost of missed bankruptcies) at the expense of some extra false positives (non‐bankrupt companies incorrectly flagged). Depending on business requirements—where catching every possible bankruptcy may outweigh occasional false alarms—this trade‐off is acceptable.

**Precision, Recall, and F1 Scores**  
From `models/rf_results`, we report the following metrics on the 20 % hold‐out set:
- **Bankrupt Class (1):**  
  - Precision₁ = 0.28 (of all companies flagged as bankrupt, 28 % actually were bankrupt)  
  - Recall₁ = 0.81 (model identifies 81 % of true bankruptcies)  
  - F1₁ = 0.42 (harmonic mean of precision₁ & recall₁)  
- **Non-bankrupt Class (0):**  
  - Precision₀ = 0.97 (of all companies predicted non-bankrupt, 97 % truly were non-bankrupt)  
  - Recall₀ = 0.93 (model identifies 93 % of true non-bankrupt firms)  
  - F1₀ = 0.95  

The high recall₁ (0.81) confirms that our primary goal—**catching as many bankruptcies as possible**—is met. Although Precision₁ is lower (0.28), this is a known trade-off when optimizing for recall in imbalanced settings. Meanwhile, the non-bankrupt class maintains strong precision and recall (> 0.90), indicating the model does not over-flag healthy companies excessively.

**AUC (ROC) Curve**  
The ROC curve (`models/rf_results/roc_curve.png`) plots true positive rate vs. false positive rate at various thresholds. Our model achieves an **AUC of 0.95**, demonstrating excellent discriminative ability: it can rank bankrupt companies above non-bankrupt ones with high confidence. A steep ROC curve near the top-left corner indicates that, even before threshold tuning, the underlying classifier is strong at separating classes.

**Precision-Recall Curve**  
In highly imbalanced datasets, the Precision-Recall curve (`models/rf_results/pr_curve.png`) is often more informative than ROC. It shows the relationship between Precision₁ and Recall₁ as the classification threshold varies. By inspecting this curve, we can select a threshold (0.27) that yields a desirable balance—maximizing recall while keeping precision above a business-defined minimum. The curve’s shape (high area under PR curve) further confirms robust performance on the minority class.

**Learning Curve**  
The learning curve (`models/rf_results/learning_curve.png`) plots model performance on both training and validation sets as the number of training samples increases. Early in training (20 % of data), a gap between training accuracy (near 1.0) and validation accuracy (around 0.92) indicates slight overfitting. As more data is added, both curves converge to ~ 0.95, reflecting that additional samples help the model generalize better. This suggests that gathering more labeled examples (especially bankrupt cases) could further reduce overfitting gaps.

**Feature Importance**  
The feature importance plot (`models/rf_results/feature_importance.png`) ranks the top predictors as determined by mean decrease in Gini impurity. The three most influential features are:  
1. **Borrowing dependency**  
2. **Total debt / Total net worth**  
3. **Net Income to Total Assets**  

These financial ratios align with domain knowledge: highly leveraged or low‐income companies are more likely to become insolvent. By identifying the most predictive features, stakeholders can prioritize monitoring these ratios for early warning signs.

---

Overall, these evaluation metrics and visualizations demonstrate that our Random Forest model, tuned for recall, effectively identifies a majority of bankrupt companies (81 % recall) while maintaining high overall discriminative power (AUC = 0.95). Future improvements could include exploring additional resampling strategies, tuning other hyperparameters, or integrating new features to push recall even higher without drastically harming precision.
