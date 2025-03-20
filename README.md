## Spam SMS Detection


# Overview
This project is a Spam SMS Detection System using Natural Language Processing (NLP) and Machine Learning. It classifies SMS messages as either Spam (unwanted/promotional messages) or Ham (legitimate messages). The model is trained using a dataset of labeled SMS messages and employs Naïve Bayes Classifier for text classification.

# Features
- Preprocessing: Tokenization, Stopword Removal, Stemming
- Feature Extraction using TF-IDF
- Naïve Bayes Classifier for SMS classification
- Custom SMS message prediction function
- Saved trained model for future use (.pkl files)

# Dataset
The dataset consists of labeled SMS messages:
Spam (1): Unwanted promotional or fraudulent messages.
Ham (0): Regular, legitimate messages.

# Installation
1️. Clone the repository
2. Install dependencies
3. Run the Jupyter Notebook
Open the notebook and execute the cells to train the model.

# Usage
1️. Train the Model
Open and run the Jupyter Notebook (Spam_sms_detection.ipynb).
The dataset will be loaded, preprocessed, and classified using Multinomial Naïve Bayes.

2. Predict Custom SMS Messages
You can classify new SMS messages using the trained model

These files can be loaded into other Python applications for SMS classification.

# Model Performance
The model is evaluated using:
Accuracy Score
Confusion Matrix
Classification Report (Precision, Recall, F1-Score)

# Dependencies
Ensure you have the following installed:
Python 3.x
pandas
numpy
nltk
scikit-learn
matplotlib
seaborn


# This project is open-source and free to use. Feel free to contribute!
# For improvements or suggestions, feel free to open an issue or a pull request! 

