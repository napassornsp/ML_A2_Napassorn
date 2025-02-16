# ML_A2_Napassorn - Car Price Prediction Web Application

Welcome to the ML_A2_Napassorn repository! This web application predicts car prices using advanced machine learning models. It integrates various car features to provide estimates using both a standard and an enhanced predictive model. Follow the instructions below to set up the application and start predicting car prices today!

## Table of Contents

- [Project Setup Instructions](#project-setup-instructions)
- [Application Guide](#application-guide)

## Project Setup Instructions

To set up and deploy the project on your local machine, please follow these steps:

1. **Install Prerequisites**:
    - Docker: [Install Docker](https://docs.docker.com/get-docker/)
    - Git: [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

2. **Clone the Repository**:
    ```bash
    git clone https://github.com/napassornsp/ML_A2_Napassorn.git
    ```

3. **Navigate to the Project Directory**:
    ```bash
    cd ML_A2_Napassorn/
    ```

4. **Deploy the Website Using Docker**:
    ```bash
    docker-compose up
    ```

5. **Access the Application**:
    For the local version : Open [http://localhost:600](http://localhost:600) in your web browser to launch the application.
   For the live version : Open https://st124949.ml.brain.cs.ait.ac.th/ in your web browser to launch the application.

## Application Guide

### 1. Access the Application
Enter the URL `https://st124949.ml.brain.cs.ait.ac.th/` in your browser to access the web application.
![image](https://github.com/user-attachments/assets/505d7a6b-324e-486d-9982-9886a0b80671)

### 2. Landing Page
Upon arrival, you are presented with two navigation options:
  - **Old Regression Model**: The original version of the price predictor.
  - **New Regression Model**: An updated prediction model with improved accuracy.
![image](https://github.com/user-attachments/assets/979032e2-2127-4cad-a880-7506f73acfb3)

### 3. Model Input Page
After selecting a model, you will be prompted to enter the required data to predict a car's price, such as:
  - Manufacturing Year
  - Kilometers Driven
  - Mileage
  - Engine Capacity
  - Maximum Power
Default values are provided if any data is missing, to help in estimating the car's price accurately.
![image](https://github.com/user-attachments/assets/18467abb-9c63-4143-934e-a76dd49eba89)
Old Regression Model Interface
![image](https://github.com/user-attachments/assets/c9606bdb-575b-48b4-a7d2-885422843a48)
New Regression Model Interface

### 4. Prediction Result Page
After inputting your data and clicking the "Predict Price" button, the predicted price will be displayed on this page.
 ![image](https://github.com/user-attachments/assets/d52a811c-ffb0-4321-a549-7eefe17cb580)


