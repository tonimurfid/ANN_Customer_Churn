Customer Churn Prediction using Artificial Neural Network
Overview
This project aims to predict customer churn for a bank using an Artificial Neural Network (ANN) model. The project was developed as part of the DeepLearning course on Udemy.

Dataset
The dataset used in this project is the "Churn Modelling" dataset from Kaggle. The dataset contains information about customer demographics and banking information. The dataset includes the following features:

RowNumber
CustomerId
Surname
CreditScore
Geography
Gender
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
Exited (target variable)
Model Architecture
The model architecture used in this project is an Artificial Neural Network (ANN) with the following layers:

Input Layer: The input layer takes in the customer features.
Hidden Layers: The model has 2 hidden layers with 64 and 32 neurons, respectively, using ReLU activation.
Output Layer: The output layer has a single neuron with a sigmoid activation function to predict the probability of customer churn.
Training and Evaluation
The model was trained using the following hyperparameters:

Optimizer: Adam
Loss Function: Binary Cross-Entropy
Epochs: 100
Batch Size: 32
The model was evaluated using the following metric:

Accuracy
Results
The final model achieved an accuracy of 0.8615 on the test set, indicating that the model is able to accurately predict customer churn.

Usage
To run the project, follow these steps:

Clone the repository: git clone https://github.com/tonimurfid/ANN_Customer_Churn.git
Install the required dependencies: pip install numpy tensorflow scikit-learn matplotlib
Run the main script: python ann_customer_churn.py
Dependencies
Python 3.7+
NumPy
Pandas
TensorFlow
Keras
Scikit-learn
Matplotlib
Future Improvements
Explore other neural network architectures, such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs), to potentially improve model performance.
Incorporate additional features, such as customer interaction data or external market factors, to enhance the model's predictive capabilities.
Implement a more robust hyperparameter tuning and model selection process to further optimize the model.
Acknowledgements
Udemy DeepLearning course
Kaggle for providing the "Churn Modelling" dataset
