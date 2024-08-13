# AlzheimInsight

Welcome to **AlzheimInsight**—an advanced machine learning and Django project focused on Alzheimer's disease detection. This project aims to provide early detection of Alzheimer's disease using machine learning models and a Django-based web application.

## Project Overview

Alzheimer's disease is a progressive neurological disorder that leads to memory loss, cognitive decline, and behavioral changes. It is one of the most severe forms of dementia, impacting millions of lives worldwide. Early detection is crucial for timely intervention and management.

**AlzheimInsight** allows users to check their likelihood of having Alzheimer's disease based on various features. The platform provides a user-friendly interface for predicting and understanding the risk of Alzheimer's, helping individuals make informed decisions about their health.

## Project Components

1. **Data Exploration and Model Building**
   - Comprehensive exploratory data analysis (EDA) was conducted on a dataset of over 2,000 patients diagnosed with Alzheimer's disease.
   - The dataset includes 40+ features related to cognitive and health metrics.
   - Utilized sophisticated machine learning models and algorithms including SVM, Random Forest, KNN, and advanced boosting algorithms like AdaBoost, Gradient Boosting, XGBoost, LightGBM, and CatBoost.
   - Implemented an Artificial Neural Network (ANN) using TensorFlow.
   - Achieved a final model accuracy of 97% with precision, recall, F1 score, and ROC AUC all ranging between 93% and 95%.

2. **Django Web Application**
   - Built a Django application to provide a web interface for users to input data and receive predictions.
   - Designed the GUI using CSS, Bootstrap, and SCSS, keeping the focus primarily on the model and backend.
   - Developed API endpoints for data submission and prediction handling.

3. **Deployment**
   - The application is deployed on Railway, providing an accessible platform for users to interact with the model.

## File Structure

- **`static/`**: Contains all static files such as images, CSS, SCSS, and JS.
- **`artifacts/`**: Includes model files and preprocessing pipeline pickle files.
- **`alzheimer/`**: Main Django project folder containing `settings.py`.
- **`api/`**: Django app for handling POST requests and making predictions.
- **`mainapp/`**: Django app used for routing.
- **`templates/`**: Contains the GUI made using Django templates.
- **`requirements.txt`**: Lists all necessary Python packages for the project.

## Data and Code

- **Raw Data**: The raw dataset can be downloaded from [this link](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset).
- **Model Building Process**: The entire model-building process is detailed in [this Kaggle notebook](https://www.kaggle.com/code/faizkhan7/alzheimer-s-disease-analysis-and-predictions). If you find it useful, please give it a like!
- **Source Code**: The complete source code for the project can be found [here](https://github.com/faizkhan77/AlzheimInsight).

## Deployment

- **Live Application**: Check out the deployed project at [AlzheimInsight on Railway](https://alzheiminsight.up.railway.app/).

## Contributing

Feedback and suggestions for improvement are welcome! If you find any areas for improvement or have any questions, feel free to reach out.

## Note

Please note that while the application focuses on model accuracy and backend functionality, the user interface is kept simple to prioritize the core functionality of the model and its predictions.
