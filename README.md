# Disaster Response Pipeline Project

### What is about this project:
[Figure Eight](https://appen.com/) has collaborated with Udacity to develop the data engineering skills to provide a service which can be 
useful to classify a message written during a disaster in 36 possible categories through a machine learning model
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

