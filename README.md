
# MBTI Personality Dataset – Exploratory Data Analysis

## Overview

This project explores the **MBTI Personality Dataset** through **Exploratory Data Analysis (EDA)** to understand patterns in personality types and written posts.

The dataset contains posts from users along with their personality types based on the **Myers–Briggs Type Indicator (MBTI)**.
The goal of this analysis is to explore the structure of the dataset, identify distributions, and investigate potential relationships between personality types and language patterns.

---

## Dataset

Source: Kaggle MBTI Personality Dataset

The dataset contains **8,675 users** and two main features:

| Feature | Description                                               |   |   |   |
| ------- | --------------------------------------------------------- | - | - | - |
| `type`  | The MBTI personality type of the user (16 possible types) |   |   |   |
| `posts` | A collection of posts written by the user separated by `  |   |   | ` |

Example personality types:

* INTJ
* INFP
* ENFP
* ISTJ

---

## Objectives

The main objectives of this analysis are:

* Understand the distribution of MBTI personality types
* Explore characteristics of user posts
* Analyze text patterns within the dataset
* Identify insights that could help build machine learning models

---

## Exploratory Data Analysis

### Dataset Overview

* Total users: **8,675**
* Number of personality types: **16**
* Each row represents a user with approximately **50 posts**

### Key Questions

During the analysis, the following questions were explored:

1. What personality types appear most frequently in the dataset?
2. Is the dataset balanced across personality types?
3. What is the distribution of introverted vs extroverted personalities?
4. How long are the posts on average?
5. Do different personality types write differently?
6. What are the most common words used in the posts?

---

## Visualizations

The following visualizations were used to explore the dataset:

* Personality type distribution (Bar Chart)
* Introvert vs Extrovert comparison
* Post length distribution (Histogram)
* Post length by personality type (Boxplot)
* Word frequency analysis
* Word cloud of most common words

---

## Insights

Some observations from the analysis include:

* Certain personality types appear more frequently than others.
* The dataset is slightly imbalanced across personality categories.
* Most posts vary significantly in length.
* Language usage patterns may differ between personality types.

---

## Tools Used

* Python
* Pandas
* Matplotlib
* Seaborn


---

## Future Work

Possible extensions of this project include:

* Text preprocessing and feature extraction
* Applying TF-IDF or Bag-of-Words
* Building a machine learning model to predict MBTI personality types from text
* Evaluating classification performance

---

## Conclusion

This exploratory analysis provides a foundational understanding of the MBTI dataset and highlights patterns that could support future personality prediction models using text data.
