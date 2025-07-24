# Gender and Age Prediction

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-green.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project focuses on building a deep learning model to accurately predict gender and age from facial images. The model is trained and evaluated using the UTKFace dataset.

## Features
Age Prediction: Predicts the age of individuals from facial images.
Gender Prediction: Classifies the gender (Male/Female) of individuals from facial images.
Data Preprocessing: Includes steps for loading, parsing, and preparing image data and corresponding labels.
Deep Learning Model: Utilizes a Convolutional Neural Network (CNN) built with Keras and TensorFlow for predictions.
Exploratory Data Analysis (EDA): Visualizations and insights into the UTKFace dataset.

## Dataset
The project utilizes the UTKFace Dataset, which consists of over 20,000 face images with annotations for age, gender, and ethnicity. The images are labeled in the format of [age]_[gender]_[ethnicity]_[date&time].jpg.
Age: Integer from 0 to 116, representing the age.
Gender: 0 for Male, 1 for Female.
Ethnicity: 0 for White, 1 for Black, 2 for Asian, 3 for Indian, and 4 for Others.
The dataset is expected to be located at /kaggle/input/utkface-new/UTKFace/.

## Technologies Used
The project is developed using Python and leverages popular libraries for data science and deep learning:

pandas: For data manipulation and analysis.
numpy: For numerical operations.
os: For interacting with the operating system (e.g., file processing).
matplotlib.pyplot: For creating static, animated, and interactive visualizations.
seaborn: For statistical data visualization.
Pillow (PIL): For opening, manipulating, and saving many different image file formats.
tensorflow: An open-source machine learning framework.
keras: A high-level neural networks API, running on top of TensorFlow. Used for building and training the CNN model.

load_img: For loading images directly into NumPy arrays.
Sequential, Model: For defining the neural network architecture.
Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input: Various layers used in the CNN.

tqdm.notebook: For displaying progress bars while loading images.

## Installation
Clone the repository:
git clone https://github.com/your-username/Gender-and-Age-prediction.git
cd Gender-and-Age-prediction

Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required packages:
pip install pandas numpy matplotlib seaborn pillow tensorflow keras tqdm

Download the Dataset:
The UTKFace dataset needs to be downloaded and placed in the appropriate directory. If you are running this on Kaggle, the path /kaggle/input/utkface-new/UTKFace/ will likely be automatically configured. For local execution, ensure the dataset is placed at a path accessible to the BASE_DIR variable in the notebook.

