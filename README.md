# Disaster Response Pipeline Project

### What is about this project:
[Figure Eight](https://appen.com/) has collaborated with Udacity to develop some data engineering skills to provide a service which can be 
useful to classify messages written during a disaster in 36 possible categories through a machine learning model
and send some messages to the respective disaster relief agency.

In these files you can find some of the resources to build that web page and prove with some aditional data about how is the performance of the model in the classification task.


### How to run this project:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

It can be valuable to create a new environment to run the project in Anaconda. All necessary packages are included in the file enviroment.yml.
To run this project in a new enviroment, you can use the tutorial in [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) using this file.

### Packages for the project

All packages you need to run the project are included in the file enviroment.yml.

### Files in the repository

Here's the file structure of the project:

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md

'App' directory contains all files to render the page and run the Python scripts.
'Data' directory contains all data used to train the model, after the cleaning and standarization of data and saved in a .db file.
'Models' directory contains the classifier (with extension .pkl) and the Python script to generate the classifier (in case you want to change the data or the model).

