# Machine Learning-Based Intrusion Detection System

## 🔒 Overview
This project implements an Intrusion Detection System (IDS) using various Machine Learning and Neural Network algorithms. The system is designed to monitor and analyze network traffic to detect potential security breaches and suspicious activities. Unlike traditional signature-based IDS, our approach leverages machine learning to identify patterns and detect anomalies that may indicate new or unknown attacks.

## 🎯 Features
- Binary and Multi-class classification of network threats
- Support for multiple ML algorithms:
  - Linear Support Vector Machine (LSVM)
  - Quadratic Support Vector Machine (QSVM)
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Multi-Layer Perceptron (MLP)
  - Long Short-Term Memory (LSTM)
- Comprehensive data preprocessing pipeline
- Model performance evaluation and comparison
- Support for NSL-KDD dataset

## 📊 Dataset
The project uses the NSL-KDD dataset, an improved version of the KDD'99 dataset. Key features include:
- No redundant records
- Balanced distribution
- Different difficulty levels
- Standardized format
- 41 features plus one label for classification

## 🛠️ Technical Implementation

### Data Preprocessing
1. **Data Cleaning**
   - Column name definition
   - Label value standardization
   - Attack label categorization

2. **Feature Engineering**
   - Normalization
   - One-hot encoding
   - Feature extraction for both binary and multi-class classification

### Classification Types
1. **Binary Classification**
   - Normal
   - Abnormal (Intrusion)

2. **Multi-Class Classification**
   - Normal
   - DoS (Denial of Service)
   - Probe
   - R2L (Remote to Local)
   - U2R (User to Root)

## 📈 Results

### Binary Classification Performance
- Random Forest: 99.25% accuracy (Best performing)
- KNN: 98.84% accuracy
- High F1, precision, and recall scores across models

### Multi-Class Classification Performance
- Random Forest: 99.16% accuracy (Best performing)
- KNN: 98.63% accuracy
- Strong performance across all attack categories

## 🚀 Getting Started

### Prerequisites
```bash
# Required Python packages
numpy
pandas
scikit-learn
tensorflow
keras
```

### Installation
```bash
git clone https://github.com/yourusername/ids-project.git
cd ids-project
pip install -r requirements.txt
```

### Usage
1. Download the NSL-KDD dataset
2. Run data preprocessing
3. Train the models
4. Evaluate performance
5. Test with new data

## 📊 Model Testing
- Uses KDDTest+ dataset for evaluation
- Includes adaptation for missing columns
- Performance metrics for both binary and multi-class classification
- Comprehensive evaluation across all implemented algorithms

## 🔮 Future Improvements
- PCAP file support with feature extraction
- Real-time traffic analysis
- Enhanced feature engineering
- Support for additional ML algorithms
- Improved handling of zero-imputed columns

## 📝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## ✨ Authors
- Oussama BENDADA

## 📄 License
This project is licensed under the MIT License - see the LICENSE.md file for details

## 🙏 Acknowledgments
- NSL-KDD dataset creators
- Anthropic research team for ML insights
- Open source community

## 📚 References
- NSL-KDD Dataset Documentation
- Machine Learning for Cybersecurity
- Neural Networks in Network Security
