# **Intrusion Detection System Using Machine Learning and Neural Networks**

Developed by: **Oussama BENDADA**  
---

## **Project Overview**
This project presents an advanced Intrusion Detection System (IDS) utilizing Machine Learning (ML) and Neural Network techniques to monitor and analyze network traffic, identifying both known and novel cyber threats. Unlike traditional IDSs that rely on rule-based detection, this system leverages ML models to dynamically learn from data, adapt to new attack patterns, and improve detection accuracy over time. The project is based on the **NSL-KDD** dataset, a widely used benchmark for intrusion detection, and includes comparative analysis across multiple ML and neural network algorithms to evaluate their performance in both binary and multi-class classification tasks.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Methodology](#methodology)
   - [Dataset](#dataset)
   - [Data Classification](#data-classification)
   - [Data Pre-Processing](#data-pre-processing)
   - [Algorithms and Models](#algorithms-and-models)
   - [Model Training](#model-training)
3. [Comparative Analysis](#comparative-analysis)
4. [Testing and Evaluation](#testing-and-evaluation)
5. [Challenges and Improvements](#challenges-and-improvements)
6. [Conclusion](#conclusion)

---

## **1. Introduction**

An **IDS** based on Machine Learning is a cybersecurity tool designed to monitor network traffic and system activity to identify suspicious patterns or unauthorized behavior. ML-based IDSs go beyond traditional IDSs, which detect threats by matching predefined rules and signatures, by using algorithms that can learn from data to detect unknown threats and adapt to evolving attack methods.

**Key Objectives**:
- Build an effective IDS using machine learning techniques.
- Explore and compare the effectiveness of different ML and neural network algorithms.
- Develop a flexible system that performs well on binary (normal vs. attack) and multi-class classification (specific attack types) tasks.

**Key Components**:
- **Data Preprocessing**: Ensuring the data is clean, normalized, and properly formatted.
- **Model Training**: Implementing various ML and neural network models for classification.
- **Performance Evaluation**: Testing and comparing models based on accuracy, precision, recall, and F1-score.

---

## **2. Methodology**

### **2.1 Dataset**
The **NSL-KDD** dataset, an improved version of the original KDD'99 dataset, is used to train and test our models. It includes:
- **KDDTrain+.txt**: Used for training, containing 41 features and labels indicating attack types.
- **KDDTest+.txt**: Used for testing, modified with additional columns to match the training dataset and ensure feature consistency.

### **2.2 Data Classification**
- **Binary Classification**: Categorizes network activity as either **normal** or **intrusion**.
- **Multi-Class Classification**: Classifies intrusions into four main categories:
  - **DoS (Denial of Service)**
  - **Probe**
  - **R2L (Remote-to-Local)**
  - **U2R (User-to-Root)**

### **2.3 Data Pre-Processing**
Data preprocessing is essential for preparing the dataset for ML training. It involves:
1. **Cleaning**: Loading data, labeling columns, removing irrelevant features, and categorizing attack types.
2. **Normalization**: Scaling numeric features to a standard range, which helps improve model convergence and performance.
3. **One-Hot Encoding**: Transforming categorical attributes into a binary format, creating separate binary features for each category.
4. **Feature Extraction**: Selecting the most relevant features for model training to improve accuracy and reduce computational complexity.

### **2.4 Algorithms and Models**
Various ML and neural network algorithms were selected for their strengths in handling different types of data and patterns. These include:

- **Support Vector Machine (SVM)**:
  - **Linear SVM (LSVM)**: Effective for high-dimensional data and binary classification.
  - **Quadratic SVM (QSVM)**: Uses a polynomial kernel for non-linear relationships.
- **K-Nearest Neighbors (KNN)**: Classifies data points based on the majority class of their nearest neighbors.
- **Decision Trees and Random Forests**:
  - **Decision Trees**: Hierarchical model that splits data based on feature values.
  - **Random Forests**: An ensemble of decision trees providing robustness and accuracy.
- **Neural Networks**:
  - **Multi-Layer Perceptron (MLP)**: A neural network with multiple layers capable of learning complex non-linear patterns.
  - **Long Short-Term Memory (LSTM)**: An RNN that excels in learning from sequential data.

### **2.5 Model Training**
Each model was trained on the NSL-KDD dataset for both binary and multi-class classifications. Parameters were fine-tuned, and hyperparameters were optimized to maximize accuracy and other performance metrics.

---

## **3. Comparative Analysis**

**Binary Classification**:
- **Top Performers**: Random Forest and KNN achieved high accuracy rates of **99.25%** and **98.84%**, respectively.
- **Evaluation Metrics**: F1-score, precision, recall, and accuracy were used to assess model effectiveness.

**Multi-Class Classification**:
- **Top Performer**: Random Forest maintained the highest accuracy, with an accuracy rate of **99.16%**, displaying robustness across all classes.

---

## **4. Testing and Evaluation**

### **4.1 Test Dataset**
To validate our models, we used the **KDDTest+.txt** dataset. Missing columns were added to match the training data structure, ensuring the model could interpret and process the data accurately. This adjustment was crucial to maintain feature consistency and ensure reliable results.

### **4.2 Performance Impact**
Adding missing columns to the test dataset introduced noise, leading to a significant drop in model accuracy (approximately 20% reduction from initial values of 95-96% to 70-78% across all models). This highlights the importance of dataset consistency in ML model performance.

---

## **5. Challenges and Improvements**

### **Current Challenges**
- **Feature Consistency**: Modifying the test dataset structure impacted accuracy.
- **Noise Introduction**: Zero-imputed columns affected model interpretability and increased error rates.

### **Future Improvements**
- **PCAP Integration**: Enhance the system by supporting **PCAP file analysis** for raw network data, adding flexibility in real-world applications.
- **Feature Optimization**: Implement advanced feature selection to minimize noise and improve accuracy.
- **Model Fine-Tuning**: Experiment with hyperparameter tuning for neural network models to handle complex data structures more effectively.

---

## **6. Conclusion**

This ML and neural network-based IDS demonstrates strong potential for adaptive intrusion detection by combining high-performance models with versatile data preprocessing techniques. Although traditional IDSs are efficient for known threats, our ML-based approach proves more effective in detecting novel, sophisticated attacks. Random Forest stood out as the top model, showing a balance of accuracy, robustness, and interpretability, making it suitable for both binary and multi-class tasks.

---
