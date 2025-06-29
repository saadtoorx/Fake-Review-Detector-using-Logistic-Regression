# 🔍 Fake Review Detector using Logistic Regression

Detect whether a review is **genuine** ✅ or **fake** ❌ using natural language processing and machine learning. This project leverages TF-IDF and logistic regression to identify deceptive content in user reviews.

---

## 🧠 About the Project

This project demonstrates a machine learning workflow to classify product or service reviews as **fake** or **genuine**. It uses TF-IDF to vectorize text, trains a Logistic Regression classifier, and includes real-time review prediction along with visual evaluation through a confusion matrix.

---

## 🚀 Features

- 🧹 Clean and preprocess text data using TF-IDF
- 🤖 Train a Logistic Regression model for binary classification
- 📊 Evaluate model performance with Accuracy, Confusion Matrix, and Classification Report
- 💬 Real-time prediction for user-provided review text
- 📸 Includes 2 output images + confusion matrix visualization

---

## 🛠️ Tech Stack

- Python 3.x  
- pandas  
- scikit-learn  
- matplotlib  
- seaborn  

---

## 📁 Project Structure

```
Fake-Review-Detector/
├── review_detector.py            # Main script
├── review_detector.ipynb         # Jupyter notebook version
├── reviews.csv                   # Labeled dataset of reviews
├── images/
│   ├── confusion_matrix.png      # Confusion matrix plot
│   ├── output_sample_1.png       # Sample prediction image
│   └── output_sample_2.png       # Sample prediction image
├── README.md                     # Project documentation
├── requirements.txt              # Required dependencies

```

---

## 💻 How to Run

1. **Clone the Repository**

```bash
git clone https://github.com/saadtoorx/Fake-Review-Detector.git
cd Fake-Review-Detector
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Script**

```bash
python review_detector.py
```

4. **Try it Out**

When prompted, enter a review in the terminal. The model will respond with a prediction:  
🟢 **Genuine** or 🔴 **Fake**

---

## 📷 Visual Output

- ✅ Sample classification results  
- 📊 Confusion matrix with labeled axes

---

## 🧾 License

This project is licensed under the [MIT License](LICENSE)

---

## 👤 Author

Made with 💡 by [@saadtoorx](https://github.com/saadtoorx)  
If you like it, ⭐ the repo and feel free to fork!

