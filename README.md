> A repository demonstrating end-to-end development of a bankruptcy prediction pipeline (Random Forest & XGBoost), including cloud feasibility analysis, data preprocessing, feature selection, model training, and evaluation.  
>
> Coursework for LD7187: ML on Cloud

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)  
2. [Cloud Feasibility Study](#cloud-feasibility-study)  
3. [Dataset Description](#dataset-description)  
4. [Data Preprocessing](#data-preprocessing)  
   - [Handling Duplicates & Missing Values](#handling-duplicates--missing-values)  
   - [Outlier Treatment](#outlier-treatment)  
   - [Feature Scaling](#feature-scaling)  
   - [Resampling (Imbalance Handling)](#resampling-imbalance-handling)  
5. [Feature Selection](#feature-selection)  
6. [Model Selection & Training](#model-selection--training)  
   - [Random Forest](#random-forest)  
   - [XGBoost](#xgboost)  
   - [Cross‐Validation](#cross-validation)  

---

## 🚀 Project Overview

The primary goal of this project is to build and deploy a robust machine learning pipeline that predicts whether a company will go bankrupt, using historical financial ratios and other firm‐level features. We explore two tree‐based algorithms—Random Forest and XGBoost—tuned to maximize recall on an imbalanced target (bankrupt vs. non-bankrupt). We also conduct a cloud feasibility study comparing AWS SageMaker, Azure Machine Learning, and Google Vertex AI, assessing each on performance, cost, security, and ease of implementation. :contentReference[oaicite:1]{index=1}

Key highlights:

- **Imbalanced dataset** (6,665 companies; 96.71 % non-bankrupt, 3.29 % bankrupt)   
- **Feature selection** via an embedded Random Forest method, reducing from 95 features to the top 30 most predictive :contentReference[oaicite:3]{index=3}  
- **Two tree-based models**:  
  - Random Forest (optimized for recall, tuned to 80 trees with a 0.27 probability threshold)   
  - XGBoost (tuned to 300 trees, learning rate = 0.2, max depth = 7, probability threshold = 0.063)   
- **Evaluation metrics**: precision, recall, F1, AUC (both models achieve AUC > 0.94 on test)   
- **Cloud feasibility**:  
  - AWS SageMaker: automatic scaling, cluster management, cost controls  
  - Azure ML: Kubernetes/Batches, real-time performance monitoring  
  - Google Vertex AI: BigQuery integration, real-time data streaming  
  :contentReference[oaicite:7]{index=7}  

---

## ☁️ Cloud Feasibility Study

Before deploying any ML model, we must understand how each major cloud provider supports (and charges for) scalable, secure ML workloads. Below is a high-level summary:

### 1. AWS SageMaker
- **Computing Performance & Scalability**:  
  Automatically scales CPU/GPU clusters; shuts down idle endpoints to save cost.  
- **Cost Efficiency**:  
  Detailed cost analysis dashboards; reserved/spot instances can cut costs by up to 70 %.  
- **Security**:  
  Comprehensive compliance (ISO 27001, SOC 1/2/3, PCI DSS, GDPR). Built-in Financial Services Compliance program for regulated data.  
- **Ease of Implementation**:  
  Integrated with AWS Glue for preprocessing, CloudWatch for monitoring, Lambda for event-driven triggers. :contentReference[oaicite:8]{index=8}

### 2. Azure Machine Learning
- **Computing Performance & Scalability**:  
  Uses Kubernetes/Azure Batch to dynamically scale. Monitors real-time metrics and auto-scales on spikes.  
- **Cost Efficiency**:  
  Reserved instances alert when budget thresholds are reached; pre-purchased compute can yield 40–60 % savings if committed 1–3 years.  
- **Security**:  
  Azure Policy/Blueprint ensures continuous compliance.  
- **Ease of Implementation**:  
  Deep integration with Azure DevOps (CI/CD pipelines), Azure Data Factory connectors for ETL/transform. :contentReference[oaicite:9]{index=9}

### 3. Google Vertex AI
- **Computing Performance & Scalability**:  
  Automated scaling on demand; handles large batches with minimal latency.  
- **Cost Efficiency**:  
  Sustained-use discounts; long-term commitments can save 40–60 %.  
- **Security**:  
  Security Command Center for threat detection, VPC Service Controls, and out-of-the-box encryption.  
- **Ease of Implementation**:  
  Native BigQuery/Dataflow integration for real-time data pipelines. :contentReference[oaicite:10]{index=10}

---

## 🗃️ Dataset Description

- **Source**: Historical financial dataset containing 6,665 rows (companies) and 96 total attributes (including the target variable “Bankrupt?”).  
- **Target**: `Bankrupt?` (0 = non-bankrupt, 1 = bankrupt). Distribution: 6,446 of 6,665 (96.71 %) are non-bankrupt; 219 of 6,665 (3.29 %) are bankrupt.   
- **Features**: Initially 95 numeric columns (various financial ratios, flags, and binary indicators).  
  - Example features (top correlations with bankruptcy):  
    - **Positively correlated**: `Total debt/Total net worth` (ρ = 0.220), `Debt ratio %` (ρ = 0.219).  
    - **Negatively correlated**: `Persistent EPS in the Last Four Seasons` (ρ = –0.240), `Net Income to Total Assets` (ρ = –0.237).  
  :contentReference[oaicite:12]{index=12}

---

## 🧹 Data Preprocessing

### Handling Duplicates & Missing Values
- **Duplicates**: None found.  
- **Missing Values**: Dataset contains no nulls (complete rows). :contentReference[oaicite:13]{index=13}

### Feature Cleaning
- **Zero-Variance Features**: Removed `Net Income Flag` (constant = 1). :contentReference[oaicite:14]{index=14}

### Outlier Treatment
- **Observation**: Boxplots revealed extreme outliers in many features.  
- **Decision**: Since dropping outlier rows would remove entire company records, and both Random Forest & XGBoost are robust to outliers, we retained all data. :contentReference[oaicite:15]{index=15}

### Feature Scaling
- **Rationale**: Many features already range [0, 1], but distributions are heavy-tailed. For algorithms that might underperform, we prepared a RobustScaler (IQR‐based), to be applied if needed.  
- **Implementation**: Only used if a model’s validation accuracy dropped below acceptable thresholds. :contentReference[oaicite:16]{index=16}

### Resampling (Imbalance Handling)
- **Technique**: SMOTE (Synthetic Minority Over-Sampling Technique) applied **after** train/test split (80 % train, 20 % test).  
- **Resulting Shape**: Training set grows from 5,332→10,310 rows with balanced classes (5,155 bankrupt vs. 5,155 non-bankrupt). :contentReference[oaicite:17]{index=17}  
- **Reason**: Ensures balanced positive/negative classes during training to prevent bias. :contentReference[oaicite:18]{index=18}

---

## 📊 Feature Selection

- **Method**: Embedded selection using Random Forest feature importances (i.e., train a quick RF on all 95 predictors, then rank by Gini importance).  
- **Outcome**: Top 30 features chosen (e.g., `Net Income to Total Assets` ranked highest at 0.0503; `Liability-Assets Flag` ranked lowest at 0).   
- **Motivation**: Reduces dimensionality, speeds up training, and limits overfitting.  

---

## 🤖 Model Selection & Training

We trained two tree-based classifiers: Random Forest and XGBoost. Both were tuned to optimize recall on this highly imbalanced dataset.

### Cross-Validation Strategy
- **Method**: Stratified K-Fold (k=5) on the **un-resampled** training set, preserving class ratios. Shuffle=True.  
- **Metrics**: Averaged F1 across folds (~ 0.65 for RF, ~ 0.69 for XGBoost) indicating both models handle minority prediction reasonably well. 

### Random Forest
- **Train/Test Split**: 80 % train (5,332 rows, original class proportions), 20 % test (1,333 rows).  
- **Resampling**: SMOTE applied to the 80 % train set → 10,310 rows balanced.  
- **Hyperparameters**:  
  - `n_estimators=80`  
  - `criterion='gini'` (default)  
  - Probability threshold adjusted from 0.50→0.27 (to boost recall on class 1).  
- **Cross-Validation F1**: Average ≈ 0.65 (fold scores: 0.6601, 0.6723, 0.6803, 0.6055, 0.6519).   
### XGBoost
- **Resampling**: Similarly applied SMOTE to the 80 % train set.
- **Hyperparameters**:
  - `n_estimators=300`  
  - `learning_rate=0.2` 
  - `max_depth=7`
  - `subsample=0.80 (rows), colsample_bytree=0.80 (features)`
  - `objective='binary:logistic', eval_metric='logloss'`
  - Probability threshold adjusted from 0.50→0.063.
- **Cross-Validation F1**: Average ≈ 0.69 (fold scores: 0.7138, 0.6886, 0.7240, 0.6649, 0.6734).


