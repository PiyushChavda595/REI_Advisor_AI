# 🏠 REI-Advisor AI

### Intelligent Real Estate Investment Analysis & Price Forecasting

REI-Advisor AI is a machine learning application designed to analyze property listings, identify potential real estate investment opportunities, and estimate property prices using quantitative property features.

The system uses two machine learning models to classify properties as **High Potential** or **Low Potential** and predict an estimated fair market price. The application is deployed using Streamlit and includes MLflow-based experiment tracking.

## 📌 Features

### 🏡 Investment Potential Classification

Classifies properties into two categories:

- **High Potential:** Properties priced below the locality-based benchmark.
- **Low Potential:** Properties priced above the locality-based benchmark.

The classification model uses a tuned Decision Tree Classifier.

### 💰 Property Price Prediction

Predicts an estimated property price using features such as property size, location, and amenities.

The regression model uses a tuned XGBoost Regressor.

### 📊 Property Valuation Analysis

Uses locality-level median price per square foot to identify properties that may be relatively undervalued or overpriced.

### 🛠️ Amenity-Based Feature Engineering

Converts unstructured amenity descriptions into a numerical `Amenity_Score` to represent property features quantitatively.

### 🔍 Model Explainability

Uses SHAP to analyze feature importance and understand which factors influence property valuations.

### 🧪 MLflow Experiment Tracking

Tracks model experiments, parameters, evaluation metrics, and model artifacts to support reproducibility.

### 🖥️ Interactive Streamlit Dashboard

Allows users to enter property information, explore valuation scenarios, and receive both a predicted price and an investment potential classification.

## 🏗️ System Architecture

```text
             Property Dataset
                     │
                     ▼
          Data Cleaning and EDA
                     │
                     ▼
            Feature Engineering
                     │
                     ▼
           Data Preprocessing
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
       Decision Tree     XGBoost
        Classifier       Regressor
              │             │
              ▼             ▼
       Investment        Estimated
        Potential       Property Price
              │             │
              └──────┬──────┘
                     │
                     ▼
              Streamlit App
                     │
                     ▼
             Results and Insights
```

## 🔄 Application Workflow

```text
Property Listings Dataset
          ↓
Data Cleaning
          ↓
Exploratory Data Analysis
          ↓
Feature Engineering
          ↓
Locality-Based Price Benchmarking
          ↓
Numerical and Categorical Preprocessing
          ↓
Model Training and Evaluation
          ↓
MLflow Experiment Tracking
          ↓
Save Models and Preprocessor
          ↓
Streamlit Application
          ↓
Price Prediction and Investment Classification
```

## 🤖 Machine Learning Models

The project addresses two related real estate analysis tasks: investment potential classification and property price regression.

### 1. Investment Potential Classification

**Problem Type:** Binary Classification

**Selected Model:** Tuned Decision Tree Classifier

The classifier categorizes property listings as `High Potential` or `Low Potential` using a locality-based price benchmark.

#### Target Engineering

The project calculates the median price per square foot for each locality and uses it to define the target categories:

- **High Potential:** Listings in the bottom 33% of the locality-based price distribution.
- **Low Potential:** Listings in the top 33% of the locality-based price distribution.
- **Buffer Zone:** The middle 33% is excluded to reduce ambiguity between clearly lower-priced and higher-priced properties.

This approach creates a classification task focused on identifying listings at the two ends of the relative price distribution.

### 2. Property Price Prediction

**Problem Type:** Regression

**Selected Model:** Tuned XGBoost Regressor

The regression model estimates property prices from quantitative features, including:

- Property size
- Location-related features
- Amenity score
- Other available property specifications

The report describes applying a logarithmic transformation to the price target to address its right-skewed distribution.

## 📊 Model Performance

The following results are reported in the project PDF.

### Classification Performance

| Model | Reported Metric | Result |
|---|---|---:|
| Decision Tree | Precision | 82% |
| Random Forest | Accuracy | 79% |
| XGBoost | Accuracy | 79% |

The tuned Decision Tree Classifier was selected for its speed and precision on the target investment class.

### Regression Performance

| Model | RMSE | Observation |
|---|---:|---|
| Decision Tree | 141.10 | Prone to overfitting |
| Random Forest | 140.88 | Best raw score, but exceeded available deployment memory |
| Tuned XGBoost | 140.99 | Selected for its balance of accuracy, scalability, and model size |

The regression RMSE is reported in the project's price scale, described in the report as approximately **₹141 lakhs**.

The report also states an \(R^2\) value of approximately 0.094 in its final performance discussion. This indicates that the regression model has limited explanatory power despite providing a quantitative baseline for valuation.

## 📊 Dataset and Exploratory Data Analysis

The project report describes a dataset containing approximately **250,000 property listings**.

Key findings from the exploratory analysis include:

- **Property size:** `Size_in_SqFt` was identified as the strongest reported price driver, with a correlation greater than 0.8.
- **Location:** Locality contributes to price differences beyond property size alone.
- **Amenities:** A higher amenity count showed a positive relationship with property prices.
- **Price distribution:** Property prices were heavily right-skewed, with high-value properties creating a long tail.
- **Value gap:** Listings classified as high potential tended to appear near the lower boundary of the price-versus-size relationship.

These findings informed the feature engineering and model selection strategy.

## 🧹 Data Processing and Feature Engineering

The project uses a preprocessing pipeline to prepare numerical and categorical property features.

### Locality-Based Price Benchmarking

The system calculates the median price per square foot for each locality. This benchmark is used to identify listings at the lower and upper ends of the local price distribution.

### Amenity Score

Unstructured amenity descriptions, such as pool, gym, and Wi-Fi information, are converted into a numerical `Amenity_Score`.

This allows the model to use amenity information as a quantitative feature.

### Price Transformation

The report describes using `np.log1p` to transform the price target because of its right-skewed distribution.

### Preprocessing Pipeline

A unified `ColumnTransformer` is used to maintain consistency between model training and inference.

**Numerical features:**

- Median imputation
- Log transformation for skewed features
- Standard scaling

**Categorical features:**

- Mode imputation
- One-hot encoding with `handle_unknown="ignore"`

This helps the application process supported categories that may not have appeared during training.

## 🔍 Model Explainability with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to analyze the factors influencing model predictions.

According to the report, the most influential factors for price prediction include:

1. `Size_in_SqFt`
2. Location-related features
3. Other property specifications

SHAP provides insight into how these features contribute to the model's predictions.

## 🧪 MLflow Experiment Tracking

MLflow is used to track machine learning experiments and improve reproducibility.

The report describes recording:

- Model parameters, including `learning_rate`
- Evaluation metrics
- Experiment runs
- Preprocessing and model artifacts

The trained models, preprocessors, and label encoders are serialized into `.joblib` files so the deployed application can use the saved training pipeline.

## 🖥️ Application

The project is deployed as a Streamlit web application.

According to the report, the application allows users to:

- Enter property specifications.
- Explore property valuation scenarios.
- Evaluate investment potential.
- Obtain an estimated property price.
- Receive both the price prediction and investment classification in a single interaction.

The report also describes testing scenarios such as adding a parking spot or changing the locality to observe how the predicted valuation changes.

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Web Application | Streamlit |
| Classification | Decision Tree Classifier |
| Regression | XGBoost Regressor |
| Model Comparison | Decision Tree, Random Forest, XGBoost |
| Data Processing | Pandas, NumPy |
| Preprocessing | Scikit-learn |
| Feature Engineering | Amenity Score, locality-based price benchmarking |
| Model Explainability | SHAP |
| Experiment Tracking | MLflow |
| Model Serialization | Joblib |
| Deployment | Streamlit Community Cloud |
| Version Control | Git / GitHub |

## 🚀 Setup and Deployment

The project report documents the model development workflow and Streamlit deployment but does not provide the complete dependency list, repository file structure, or verified local installation commands.

For local setup:

1. Clone the repository.
2. Review the repository's dependency files and install the required packages.
3. Ensure the trained model artifacts and preprocessing pipeline are available.
4. Follow the application's entry point and configuration to launch the Streamlit interface.

### Clone the Repository

```bash
git clone https://github.com/PiyushChavda595/REI_Advisor_AI.git
```

```bash
cd REI_Advisor_AI
```

The report identifies Streamlit Cloud as the hosting platform, but it does not provide the direct live application URL.

## 💼 Business Value

REI-Advisor AI provides a data-driven approach to property analysis by combining relative investment potential classification with estimated property price prediction.

Potential uses include:

- Helping buyers compare asking prices against locality-based benchmarks.
- Helping investors shortlist properties for further investigation.
- Helping sellers understand quantitative property valuation factors.
- Providing a consistent starting point for comparing property listings.

The model's outputs are intended to support property analysis rather than replace professional valuation or due diligence.

## ⚠️ Limitations and Considerations

### Price Prediction Accuracy

The reported regression RMSE is approximately ₹141 lakhs, and the report gives an \(R^2\) value of approximately 0.094. The price prediction therefore has substantial limitations and should be treated as a baseline estimate rather than a precise market valuation.

### Limited Property Features

The model uses available quantitative property features. Factors such as interior condition, views, and other unrecorded characteristics may influence actual market prices.

### Relative Investment Classification

The `High Potential` and `Low Potential` labels are derived from locality-level price distributions. They indicate relative price positioning within the dataset and do not guarantee investment returns.

### Deployment Constraints

The report states that the Random Forest regression model exceeded available memory during deployment. The tuned XGBoost model was selected to balance predictive performance and deployment constraints.

## 🔮 Future Scope

The project report identifies several possible improvements.

### 📍 Hyper-Local Geolocation

Integrate precise latitude and longitude information to capture micro-market price differences, such as sea views versus city views.

### 🖼️ Visual Property Analytics

Use computer vision models to analyze property listing images and estimate a property condition score.

### 📰 Market Sentiment Analysis

Incorporate news feeds, interest rates, and macroeconomic information to help account for changing market conditions.

### 🧪 Model Optimization

Continue improving model performance, feature engineering, and the balance between prediction quality and deployment resource requirements.

## 👨‍💻 Project Information

- **Project:** REI-Advisor AI
- **Full Name:** Intelligent Real Estate Investment Analysis & Price Forecasting
- **Author:** Piyush Chavda
- **Report Date:** December 12, 2025
- **Purpose:** Property price estimation and real estate investment potential classification
