# Gender and Age Prediction from Facial Images

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-green.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 💡 About the Project

This project delves into the fascinating field of computer vision to build a robust deep learning model capable of predicting both gender and age from facial images. Leveraging the power of Convolutional Neural Networks (CNNs), the model is trained on a diverse dataset to recognize subtle features indicative of age and gender. This kind of technology has applications in various domains, from targeted advertising to security and demographics analysis.

## ✨ Features

* **Accurate Age Prediction:** Predicts the approximate age of an individual.
* **Binary Gender Classification:** Classifies the gender (Male/Female) of individuals.
* **Robust Preprocessing Pipeline:** Handles image loading, parsing metadata from filenames, and preparing data for model training.
* **Deep Learning Architecture:** Employs a custom-built CNN for effective feature extraction and prediction.
* **Exploratory Data Analysis (EDA):** Provides insights into the dataset's distribution of age and gender.

## 💾 Dataset

The project utilizes the **UTKFace Dataset**, a large-scale face dataset with over 20,000 images, spanning various age, gender, and ethnicity groups. Each image filename contains annotations for:

* **Age:** `[0-116]`
* **Gender:** `[0: Male, 1: Female]`
* **Ethnicity:** `[0: White, 1: Black, 2: Asian, 3: Indian, 4: Others]`

**Dataset Path:** The project expects the dataset to be located at `/kaggle/input/utkface-new/UTKFace/`. For local execution, ensure this path is updated or the dataset is placed accordingly.

## 🛠 Technologies Used

The project is developed using Python and leverages popular libraries for data science and deep learning:

* **Python 3.x**
* **Deep Learning Frameworks:**
    * [TensorFlow](https://www.tensorflow.org/)
    * [Keras](https://keras.io/)
* **Data Manipulation & Analysis:**
    * [pandas](https://pandas.pydata.org/)
    * [NumPy](https://numpy.org/)
* **Image Processing:**
    * [Pillow (PIL)](https://python-pillow.org/)
* **Visualization:**
    * [Matplotlib](https://matplotlib.org/)
    * [Seaborn](https://seaborn.pydata.org/)
* **Utilities:**
    * `os` (for file system operations)
    * `tqdm` (for progress bars)

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Make sure you have Python 3.x installed.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/Gender-and-Age-prediction.git](https://github.com/your-username/Gender-and-Age-prediction.git)
    cd Gender-and-Age-prediction
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required libraries:**

    ```bash
    pip install -r requirements.txt
    ```
    *Note: You might need to create a `requirements.txt` file first. See the "Usage" section for how to generate it.*

4.  **Download the UTKFace Dataset:**
    Download the dataset from its official source (e.g., Kaggle if available) and place it such that the `UTKFace` directory is accessible at the path specified in the notebook (or adjust the path in the notebook).

## 🏃‍♀️ Usage

1.  **Generate `requirements.txt` (if you don't have one):**
    If you've already run the notebook and installed dependencies, you can create a `requirements.txt` file using:

    ```bash
    pip freeze > requirements.txt
    ```

2.  **Run the Jupyter Notebook:**

    ```bash
    jupyter notebook Gender-and-Age-prediction.ipynb
    ```

    This will open the Jupyter interface in your browser. You can then execute the cells in the `Gender-and-Age-prediction.ipynb` notebook step-by-step to:
    * Load and preprocess the data.
    * Perform EDA.
    * Build and train the CNN models for age and gender prediction.
    * Evaluate model performance.

## 📁 Project Structure
