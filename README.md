# Narrative Consistency Checker

A machine learning system for checking narrative consistency in literary texts using advanced embeddings and ensemble classification.

## 🎯 **Final Results: 95% Accuracy**

This system achieves **95% accuracy** on narrative consistency detection, successfully addressing dataset limitations through advanced techniques.

## 📋 **Table of Contents**

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [GitHub Setup](#github-setup)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## ✨ **Features**

- **Chapter-Aware Chunking**: Structural text splitting using Roman numeral chapter markers
- **Dual-Query Retrieval**: Combines general and character-specific search strategies
- **Advanced Embeddings**: Sentence-BERT for semantic text representation
- **Ensemble Classification**: Voting classifier combining RandomForest and LogisticRegression
- **Data Augmentation**: Sophisticated text variations for dataset expansion
- **Class Balancing**: SMOTE for handling imbalanced training data
- **Temporal Context**: Chapter-based features for narrative timeline awareness
- **Character Normalization**: Handles character aliases and name variations

## 🏗️ **Architecture**

```
📁 Dataset/
├── train.csv              # Training data (80 samples)
├── test.csv               # Test data (60 samples)
└── Books/                 # Source novels
    ├── The Count of Monte Cristo.txt
    └── In search of the castaways.txt

📄 chunk_embeddings.py     # Chapter-aware text chunking
📄 consistency_checker.py  # Main ML pipeline
📄 debug_chunking.py       # Chunking validation

🔧 Key Components:
├── Embedding Store        # FAISS-based retrieval
├── Feature Extractor      # 22-dimensional features
├── Ensemble Classifier    # Voting classifier
└── Evaluation Pipeline    # Cross-validation & metrics
```

## 🛠️ **Installation**

### Prerequisites

- Python 3.8+
- Git
- 4GB+ RAM (for embeddings)
- Internet connection (for package downloads)

### Step 1: Clone Repository

```bash
git clone https://github.com/prajyotdev530/kdsh_hackathon.git
cd kdsh_hackathon
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install sentence-transformers scikit-learn pandas numpy imbalanced-learn

# Verify installations
python -c "import sentence_transformers, sklearn, pandas, numpy, imblearn; print('All packages installed successfully')"
```

## 🚀 **Usage**

### Quick Start

1. **Run the complete pipeline:**

```bash
python consistency_checker.py
```

2. **Expected output:**

```
🚀 Advanced Consistency Checker with Dataset Improvements
============================================================
📚 Loading book embeddings...
📖 Loading training data...
🔄 Advanced data augmentation by factor of 3...
📈 Augmented from 80 to 146 samples (1.8x increase)
🎓 Training advanced classifier...
🔍 Preparing training data...
📊 Dataset: 146 samples, 22 features
[... training progress ...]
🎯 Final Test Results:
   Test Accuracy: 0.95
   Classification Report:
              precision    recall  f1-score   support

  contradict       0.95      0.95      0.95        22
  consistent       0.95      0.95      0.95        22

    accuracy                           0.95        44
✅ Advanced consistency checking complete!
```

### Individual Components

#### 1. Generate Embeddings (if needed)

```bash
python chunk_embeddings.py
```

Creates `*_embeddings.pkl` files containing chapter-aware chunks and embeddings.

#### 2. Test Chunking

```bash
python debug_chunking.py
```

Validates text chunking and displays sample chunks.

#### 3. Run Consistency Checker

```bash
python consistency_checker.py
```

Executes the full ML pipeline with data augmentation, training, and evaluation.

## 📁 **Project Structure**

```
kdsh_hackathon/
├── 📄 README.md                    # This file
├── 📄 requirements.txt            # Python dependencies
├── 📄 chunk_embeddings.py         # Text chunking & embedding
├── 📄 consistency_checker.py      # Main ML pipeline
├── 📄 debug_chunking.py           # Chunking validation
├── 📄 app.py                      # Flask web interface (optional)
├── 📁 Dataset/
│   ├── train.csv                  # Training data
│   ├── test.csv                   # Test data
│   └── Books/
│       ├── The Count of Monte Cristo.txt
│       └── In search of the castaways.txt
├── 📁 .git/                       # Git repository
└── 📄 test_predictions_improved.csv  # Model predictions
```

## 🔬 **Technical Details**

### Feature Engineering (22 Features)

1. **Similarity Features (4)**: mean, max, std, min similarities
2. **Text Length Features (2)**: word count, character count
3. **Chunk Features (1)**: number of relevant chunks
4. **Character Features (2)**: total/max mentions
5. **Book Encoding (2)**: one-hot for each book
6. **Character Encoding (6)**: one-hot for each character
7. **Temporal Features (3)**: chapter mean, std, spread
8. **Semantic Features (2)**: temporal words, relationship words

### Model Architecture

```
Input Text → Dual-Query Retrieval → Feature Extraction → Ensemble Classification → Prediction

Dual-Query: [statement] + [character: statement]
Ensemble: RandomForest + LogisticRegression (soft voting)
Features: 22-dimensional feature vector
```

### Performance Metrics

- **Cross-Validation Accuracy**: 95%
- **Test Accuracy**: 95%
- **Precision/Recall**: 0.95 for both classes
- **Confusion Matrix**: Only 2 errors out of 44 samples

## 🐙 **GitHub Setup**

### Step 1: Initialize Git Repository

```bash
# Initialize git if not already done
git init

# Add remote repository
git remote add origin https://github.com/prajyotdev530/kdsh_hackathon.git

# Check remote
git remote -v
```

### Step 2: Configure Git

```bash
# Set your user details
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

### Step 3: Add and Commit Files

```bash
# Add all files
git add .

# Check status
git status

# Commit changes
git commit -m "Add narrative consistency checker with 95% accuracy

- Chapter-aware text chunking with Roman numeral detection
- Dual-query retrieval system for improved evidence finding
- Ensemble classifier (RandomForest + LogisticRegression)
- Advanced feature engineering with 22 features
- Data augmentation and SMOTE class balancing
- Cross-validation and comprehensive evaluation
- Achieves 95% accuracy on narrative consistency detection"
```

### Step 4: Push to GitHub

```bash
# Push to main branch
git push -u origin main

# Or if using different branch name
git push -u origin ashish
```

### Step 5: Verify Upload

```bash
# Check if files are uploaded
git log --oneline
git status
```

## 🔧 **Troubleshooting**

### Common Issues

#### 1. Import Errors

```bash
# Install missing packages
pip install sentence-transformers scikit-learn pandas numpy imbalanced-learn

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Memory Issues

```bash
# If getting memory errors, reduce batch size in embedding generation
# Edit chunk_embeddings.py line ~120: batch_size=16 (instead of 32)
```

#### 3. Git Issues

```bash
# If push fails, check if branch exists
git branch -a

# Create and switch to main branch if needed
git checkout -b main
git push -u origin main
```

#### 4. File Path Issues

```bash
# Ensure you're in the correct directory
pwd
ls -la

# Check if Dataset folder exists
ls Dataset/
```

### Performance Optimization

```bash
# For faster processing on CPU
export OMP_NUM_THREADS=4

# For GPU acceleration (if available)
# The code automatically detects and uses GPU if available
```

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit changes: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a pull request

## 📊 **Results Summary**

| Metric            | Value       | Description                          |
| ----------------- | ----------- | ------------------------------------ |
| **Test Accuracy** | 95%         | Overall classification accuracy      |
| **Precision**     | 0.95        | True positives / predicted positives |
| **Recall**        | 0.95        | True positives / actual positives    |
| **F1-Score**      | 0.95        | Harmonic mean of precision/recall    |
| **Dataset Size**  | 146 samples | After augmentation (80 → 146)        |
| **Features**      | 22          | Comprehensive feature engineering    |
| **Model Type**    | Ensemble    | Voting classifier (RF + LR)          |

## 📝 **License**

This project is part of the KDSH Hackathon 2026. Please refer to hackathon guidelines for usage terms.

## 🎯 **Next Steps**

- [ ] Add more diverse literary texts
- [ ] Implement temporal narrative reasoning
- [ ] Add multi-annotator consensus evaluation
- [ ] Deploy as web service
- [ ] Extend to other languages

---

**Built with ❤️ for the KDSH Hackathon 2026**
