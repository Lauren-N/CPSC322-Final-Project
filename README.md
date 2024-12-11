# CPSC322-Final-Project

## Project: Asthma Diagnosis Classification
Asthma is a chronic lung disease that causes inflammation and tightening of airways, making it hard to breathe.
For this project, we worked to create models to predict a patient’s asthma diagnosis using an asthma disease dataset that we obtained from kaggle.

This dataset contains patient information for 2,392 patients who are diagnosed with asthma as well as those who don’t. 
It had 29 attributes such as Age, Gender, Ethnicity, BMI, Smoking, PhysicalActivity LungFunctionFEV1, LungFunctionFVC, and others.
Our diagnosis attribute was a binary attribute so a 0 or 1, and it was the attribute we were having our models predict.

We created a K-Nearest Neighbors, Naive Bayes, and Random Forest classifier for this dataset and then selected the model we felt performed the best. 

## How to Run Our Project
To run our project: 
1. Clone our [repository](https://github.com/Lauren-N/CPSC322-Final-Project)
2. Attach to the Docker Container for our CPSC course (so you have all of the necessary packages and python versions installed)
3. Navigate to the TechnicalReport.ipynb Jupyter Notebook and run the notebook

To run and use the flask app for our project:
1. Go to this site: https://cpsc322-final-project.onrender.com/
2. On the website for BMI enter a value between 15 and 40 (can be decimal value) to represent body mass index
3. For Smoking, enter either 0 if you don't smoke or 1 if you do
4. For PhysicalActivity, enter a value between 0 and 10 for the hours you exercise per week (can be decimal value)
5. For LungFunctionFEV1, enter a value between 1.0 and 4.0 liters for your lung function for Forced Expiratory Volume in 1 second
6. Then click the "Predict Asthma Diagnosis" button to get the prediction!

## How Our Project is Organized
* There are the asthma_app.py, asthma_client.py, requirements.txt, and tree_pickler.py (which generates the tree.p file when run) files for the flask app deployment. There is also the templates folder with the index.html file for the web structure of the flask app interface.
* The classifier.py file contains the classifier classes (such as MyRandomForestClassifier)
* The myevaluation.py file contains functions for splitting the data (like train_test_split()) and evaluation metric calculation functions (like binary_precision_score()).
* The myutils.py file contains helper functions we created.
* The tests.py file contains the tests for testing our classifiers!
* The TechnicalReport.ipynb file is our Jupyter Notebook file that is used to run our project (the main part of our project that culminates all of our code). This ipynb file has both markdown and python code cells to work through all the steps of our project (from data visualization to the classification).