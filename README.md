# 🪨 Rock vs Mine Prediction

This project aims to build a **machine learning model** that predicts whether an object detected by sonar is a **Rock** or a **Mine**.  
The dataset used in this project is the **Sonar Dataset**, which contains 208 samples of sonar signals bounced off different surfaces.

---

## 📘 Overview

The project trains a **binary classification model** to distinguish between rocks and mines based on **60 numerical features** representing sonar frequency responses.

### Tasks Performed:
- Data Loading and Exploration  
- Data Preprocessing (Label Encoding, Scaling, Splitting)  
- Model Training using Logistic Regression  
- Model Evaluation (Accuracy Score)  
- Prediction on New Input Data  

---

## 🧠 Technologies Used

- **Python**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Colab Notebook**

---

## 📂 Dataset

**File:** `sonar data.csv`  
Each row in the dataset contains 60 sonar readings followed by a label:
- **R** → Rock  
- **M** → Mine  

**Dataset Details:**
- Total samples: 208  
- Features: 60  
- Label: 1 (Rock/Mine)  

---

## ⚙️ Model Workflow

1. **Data Preprocessing**
   - Load dataset using `pandas`
   - Separate features and target
   - Encode target labels (`R` → 0, `M` → 1)
   - Split into training and testing sets (typically 80–20)
   - Normalize the feature data

2. **Model Training**
   - Logistic Regression is used as a classifier.
   - Trained on the training dataset.

3. **Evaluation**
   - Evaluate accuracy on both training and test sets.
   - Compare predictions vs. actual results.

4. **Prediction**
   - Predict the output (Rock or Mine) for new input data.

---

## 📊 Example Prediction

```python
# Example input (60 values)
input_data = (0.02, 0.037, 0.04, ..., 0.005)
prediction = model.predict([input_data])
if prediction[0] == 1:
    print("Object is a Mine")
else:
    print("Object is a Rock")
```

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/anikesh-17/Rock-vs-Mine-Prediction.git
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn
   ```

3. Open Jupyter Notebook:
   ```bash
   jupyter notebook Rock_vs_Mine_Pred.ipynb
   ```

4. Run all cells to train and test the model.

---

## 🧩 Results

| Metric | Score |
|:--------|:------|
| Training Accuracy | ~83% |
| Test Accuracy | ~76% |

*(Values may vary slightly depending on random train-test splits.)*

---

## 📈 Future Improvements
- Try other classifiers (e.g., SVM, Random Forest, Neural Networks)
- Perform feature selection
- Use cross-validation for better model evaluation
- Deploy the model using Flask or Streamlit

---

## 👨‍💻 Author
**Anikesh Sharma**  
B.Tech in Computer Science | Machine Learning & AI Enthusiast  
[GitHub](https://github.com/anikesh-17)
