# Lung Cancer Detection Project

This repository contains code for detecting lung cancer using a neural network model. The model is trained on a dataset of survey responses related to lung cancer, with features such as age, gender, smoking habits, and more.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Plots and Visualizations](#plots-and-visualizations)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The goal of this project is to create a machine learning model capable of predicting the presence of lung cancer based on a set of input features. The project includes data preprocessing, model training, evaluation, and visualization of results.

## Dataset

The dataset which I have used for this project is ` lung.csv`, which contains features such as:

- Age
- Gender
- Smoking habits
- Coughing
- Wheezing
- Shortness of breath
- ... and other health-related features

The target variable is `LUNG_CANCER`, indicating the presence ('YES') or absence ('NO') of lung cancer.

## Installation

To run this project locally, please ensure you have the following installed:

- Python 3.6+
- TensorFlow 2.x
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

You can install the required packages using the following command:

```bash
pip install tensorflow pandas scikit-learn matplotlib seaborn
```
Certainly! Below is a template for a README file that you can use for your GitHub repository. It includes sections for project description, setup instructions, usage, and more.

markdown

# Lung Cancer Detection Project

This repository contains code for detecting lung cancer using a neural network model. The model is trained on a dataset of survey responses related to lung cancer, with features such as age, gender, smoking habits, and more.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Plots and Visualizations](#plots-and-visualizations)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The goal of this project is to create a machine learning model capable of predicting the presence of lung cancer based on a set of input features. The project includes data preprocessing, model training, evaluation, and visualization of results.

## Dataset

The dataset used for this project is `survey lung cancer.csv`, which contains features such as:

- Age
- Gender
- Smoking habits
- Coughing
- Wheezing
- Shortness of breath
- ... and other health-related features

The target variable is `LUNG_CANCER`, indicating the presence ('YES') or absence ('NO') of lung cancer.

## Installation

To run this project locally, please ensure you have the following installed:

- Python 3.6+
- TensorFlow 2.x
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

You can install the required packages using the following command:

```bash
pip install tensorflow pandas scikit-learn matplotlib seaborn
```
# Model Architecture

The neural network model consists of the following layers:

    Input layer
    Dense layer with 64 neurons and ReLU activation
    Dense layer with 32 neurons and ReLU activation
    Output layer with 1 neuron and sigmoid activation

The model is compiled using the Adam optimizer and binary cross-entropy loss function. It is trained for 50 epochs with a batch size of 32.
![image](https://github.com/Rama-Lekshmi/lung-cancer-detection/assets/118541549/c0fdf2fe-daf4-4126-9416-9dba20b70845)

# Process:

Data Collection: 
Secure a dataset containing relevant lung cancer prediction features, such as medical history, imaging data (e.g., CT scans), genetic markers and smoking status amongst others. They can be obtained from medical databases, research repositories or collected in collaboration with health care providers.

Data Preprocessing:
This sep may include steps like handling missing values, encoding categorical variables, and scaling numerical features.
Handle Missing Values:Determine whether there are any missing values in the dataset and choose an appropriate strategy for dealing with them through imputation or removal.
Feature Engineering: Extract significant features from the data or mold existing ones to make them more suitable for modeling purposes.
Normalize/Scale Features: This is a process of scaling numerical features so that they have similar ranges and this can help model to converge faster during training sessions.

Exploratory Data Analysis (EDA):
Explore how features are distributed and related to the target variable

Visualize data in order to see patterns/correlations

Identify Outliers/Anomalies if any

Splitting Data: The whole data set will be divided into two parts – testing set and training set

# PROGRAM:
```py
import pandas as pd

file_path = '/mnt/data/survey lung cancer.csv'
df = pd.read_csv(file_path)


print(df.head())


print(df.columns)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(df.isnull().sum())


df = df.dropna()


print(df['LUNG_CANCER'].unique())

df['LUNG_CANCER'] = df['LUNG_CANCER'].apply(lambda x: 1 if x == 'YES' else 0)


X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

X = pd.get_dummies(X, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


y_pred = (model.predict(X_test) > 0.5).astype("int32")

print(y_pred[:10])
# Save the model
model.save('lung_cancer_detection_model.h5')

from tensorflow.keras.models import load_model
loaded_model = load_model('lung_cancer_detection_model.h5')

from sklearn.metrics import classification_report, confusion_matrix


cm = confusion_matrix(y_test, y_pred)
print(cm)


cr = classification_report(y_test, y_pred)
print(cr)
```

# OUTPUT:

CONFUSION MATRIX

![lcd 1](https://github.com/Rama-Lekshmi/lung-cancer-detection/assets/118541549/2a135545-6ea2-44a0-90a9-2687f96a2d66)

CLASSSIFICATION REPORT

![lcd 2](https://github.com/Rama-Lekshmi/lung-cancer-detection/assets/118541549/1530a049-fc85-47d2-8c90-4387e344d7a5)

# CONCLUSION:
This project was developed to create predictive models for lung cancer risk using demographic data, medical data, and imaging data. The evaluation demonstrated a high level of performance which implies that this project can help healthcare providers to detect and intervene early enough in the situation.
