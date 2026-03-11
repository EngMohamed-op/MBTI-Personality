
# MBTI Personality Insights & Prediction Dashboard

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EngMohamed-op/MBTI-Personality/blob/main/EDA_project.ipynb)

A comprehensive data science project that explores personality patterns through text analysis and provides a real-time prediction demo using machine learning. This project leverages the [Kaggle MBTI Personality Dataset](https://www.kaggle.com/datasnaek/mbti-type) to identify how different personality types express themselves online.

---

## 🚀 Project Overview

This repository captures the full lifecycle of a data science project:
1.  **Exploratory Data Analysis (EDA)**: Deep dive into 8,600+ users' writing styles.
2.  **Machine Learning**: Building binary classifiers for each personality dimension.
3.  **Interactive Dashboard**: A production-ready Streamlit app for visualizing insights and testing predictions.

---

## 📊 Exploratory Data Analysis (EDA)

The core analysis is performed in `EDA_project.ipynb`. We analyzed 16 personality types across 4 dimensions: **Introversion (I) vs. Extroversion (E)**, **Intuition (N) vs. Sensing (S)**, **Thinking (T) vs. Feeling (F)**, and **Judging (J) vs. Perceiving (P)**.

### Key Insights
*   **Dataset Imbalance**: The dataset is skewed toward Introverted (I) and Intuitive (N) types, reflecting the demographic of online personality forums.
*   **Writing Style**: Introverts tend to write longer posts on average compared to Extroverts.
*   **Sentiment Correlation**: Feeling (F) types generally exhibit higher positive sentiment in their writing than Thinking (T) types.
*   **Lexical Diversity**: No significant difference in word variety was found across types, suggesting that personality influences *what* we say more than our total vocabulary size.

---

## 🤖 Machine Learning Model

The project includes an **MBTI Prediction Demo** that uses a multi-stage classification pipeline:
*   **Preprocessing**: Custom text cleaning to remove URLs, specific MBTI mentions, and special characters.
*   **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical features.
*   **Model**: Four separate **Logistic Regression** classifiers, each trained to predict one of the binary dimensions.
*   **Performance**: The model provides a "Full MBTI" prediction by combining the results of all four classifiers, along with confidence scores for each dimension.

---

## 🧠 Interactive Dashboard

The dashboard (`app.py`) provides a professional interface to explore the results:

*   **📊 Type Distribution**: Visualizes the representation of each MBTI type.
*   **📝 Post Analysis**: Deep dive into word counts and writing length by type.
*   **💬 Sentiment Analysis**: Interactive charts showing polarity across dimensions.
*   **☁️ Word Cloud**: Dynamic generation of frequent words for any selected type.
*   **🔗 Dimensions**: Side-by-side comparison of personality pairs.
*   **🤖 MBTI Prediction**: A real-time demo where you can generate random sample predictions and see the model's confidence breakdown.
*   **🔍 Explorer**: Filter and download the processed dataset.

---

## 🛠️ Technologies Used

*   **Python**: Core language
*   **Pandas & NumPy**: Data manipulation and feature engineering
*   **Scikit-learn**: Machine learning pipeline and modeling
*   **Streamlit**: Web dashboard framework
*   **Plotly**: Interactive data visualizations
*   **Matplotlib / Seaborn**: Static plots and word clouds
*   **TextBlob**: Sentiment analysis

---

## 🏃 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/EngMohamed-op/MBTI-Personality.git
cd MBTI-Personality
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
MBTI-Personality/
│
├── EDA_project.ipynb       # Original analysis and model experimentation
├── app.py                  # Production Streamlit dashboard
├── requirements.txt        # Project dependencies
├── mbti_1.csv              # Dataset (if present locally)
└── README.md               # Project documentation
```

---

## 📝 Conclusion

This project demonstrates how natural language processing (NLP) and machine learning can be used to identify subtle patterns in human personality based on written communication. The interactive dashboard makes these technical insights accessible and engaging for any user.
