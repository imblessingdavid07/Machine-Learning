### XGBoost Evaluation & Visualization

After training our XGBoost classifier (with SMOTE‐resampled data and an adjusted threshold of 0.063), we evaluate performance on the untouched 20 % test set. The following metrics and plots demonstrate how effectively the model distinguishes bankrupt vs. non‐bankrupt companies.

- **Confusion Matrix:**
 <p align="center">
  <img src="https://raw.githubusercontent.com/imblessingdavid07/Machine-Learning/main/xgb_results/ConfusionMatrix.png" alt="Random Forest Confusion Matrix" width="400" />
</p>
  - **True Positives (TP):** Out of 42 actual bankrupt cases, the model correctly flags 26 → **62 % recall** on the minority (“bankrupt”) class.  
  - **True Negatives (TN):** Out of 1,291 non‐bankrupt companies, 1,216 are correctly classified → **94 % recall** on the majority (“non‐bankrupt”) class.

- **Classification Metrics (20 % hold‐out test set):**
   <p align="center">
  <img src="https://raw.githubusercontent.com/imblessingdavid07/Machine-Learning/main/xgb_results/ClassificationReport.png" alt="Random Forest Confusion Matrix" width="400" />
</p>
  - **Bankrupt Class (1):**
    - **Precision₁ = 0.26** (of all companies flagged as bankrupt, 26 % actually were bankrupt)  
    - **Recall₁ = 0.62** (model identifies 62 % of true bankruptcies)  
    - **F1₁ = 0.36** (harmonic mean of Precision₁ & Recall₁)  
  - **Non‐bankrupt Class (0):**
    - **Precision₀ = 0.96** (of all companies predicted non‐bankrupt, 96 % truly were non‐bankrupt)  
    - **Recall₀ = 0.94** (model identifies 94 % of true non‐bankrupt firms)  
    - **F1₀ = 0.95**  

- **AUC (ROC) Curve**
   <p align="center">
  <img src="https://raw.githubusercontent.com/imblessingdavid07/Machine-Learning/main/xgb_results/ROCCurve.png" alt="Random Forest Confusion Matrix" width="400" />
</p>
The ROC curve (`xgb_results/roc_curve.png`) plots true positive rate vs. false positive rate at various thresholds. Our XGBoost model achieves an **AUC of 0.94**, indicating strong discriminative ability between bankrupt and non-bankrupt companies. A steep ROC curve near the top-left corner shows that even before threshold adjustment, XGBoost effectively separates the two classes.

**Precision-Recall Curve**  
 <p align="center">
  <img src="https://raw.githubusercontent.com/imblessingdavid07/Machine-Learning/main/xgb_results/PrecisionRecallcurve.png" alt="Random Forest Confusion Matrix" width="400" />
</p>
In imbalanced datasets, the Precision-Recall curve (`xgb_results/pr_curve.png`) is often more revealing than ROC. It illustrates the trade-off between Precision₁ and Recall₁ as the classification threshold varies. By examining this curve, we selected a threshold (0.063) that maximizes recall for the minority (bankrupt) class while maintaining acceptable precision. The high area under the PR curve confirms robust performance on detecting bankrupt companies.

**Learning Curve**  
 <p align="center">
  <img src="https://raw.githubusercontent.com/imblessingdavid07/Machine-Learning/main/xgb_results/LearningCurve.png" alt="Random Forest Confusion Matrix" width="400" />
</p>
The learning curve (`xgb_results/learning_curve.png`) shows model performance on both training and validation sets as the number of training samples increases. Early in training (20 % of data), there is a gap—training accuracy near 1.0 vs. validation around 0.88—indicating slight overfitting. As more data is used, both curves converge toward ~ 0.92, suggesting that additional samples help XGBoost generalize better. This implies that collecting more labeled bankrupt examples could further reduce overfitting.

**Feature Importance**  
 <p align="center">
  <img src="https://raw.githubusercontent.com/imblessingdavid07/Machine-Learning/main/xgb_results/FeatureImportance.png" alt="Random Forest Confusion Matrix" width="400" />
</p>
The feature importance plot (`xgb_results/feature_importance.png`) ranks the top predictors based on gain in the tree splits. The three most influential features are:  
1. **Borrowing dependency**  
2. **Net Income to Total Assets**  
3. **Debt ratio %**  

These features align with domain knowledge: companies with high leverage or low profitability are more likely to go bankrupt. By focusing on these top predictors, stakeholders can prioritize early warning indicators for financial distress.





