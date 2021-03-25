# Project: Disaster Response Pipeline

## Table of Content

- [Project Overview](#overview)
- [Project Software Stack](#stack)
  - [ETL Pipeline](#etl_pipeline)
  - [ML Pipeline](#ml_pipeline)
  - [Flask Web App](#flask)
- [Run the Project](#run)
  - [Data Cleaning](#cleaning)
  - [Training ML Classifier](#training)
  - [Run the Web App](#runapp)
- [Conclusion](#conclusion)
- [File Structure](#files)
- [Software Requirements](#sw_requirements)
- [Links](#links)


<a id='overview'></a>

## 1. Project Overview

In the “Disaster Response Pipeline” project, I will apply data engineering and machine learning to analyze disaster data provided by <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> and <a href="https://www.udacity.com/" target="_blank">Udacity</a> to build a ML classifier model that classifies disaster messages from social media and news.

The 'data' directory contains real messages that were sent during disaster events. I will create a machine learning pipeline to categorize these events so that appropriate disaster help agencies can be reached out for help.

In the project data 26248 messages with a unique id are included. Each massage will be categorized in the ML model within 36 categories.   

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


<a id='stack'></a>

## 2. Project Software Stack

The software stack of this project contains three main parts:

<a id='etl_pipeline'></a>

### 2.1. ETL Pipeline

File _/data/process_data.py_ contains data cleaning pipeline:

- Loads the 'disaster_messages' and 'disaster_categories' dataset
- Merges the two datasets in one
- Cleans the data in the combined data frame
- Stores the data in a **SQLite database “DisasterResponse.db” **

<a id='ml_pipeline'></a>

### 2.2. ML Pipeline

File _/models/train_classifier.py_ contains the machine learning pipeline:

- Loads data from the **SQLite database “DisasterResponse.db”**
- Splits the data into train and test data sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs analytics result on the test set
- Exports the final model as a pickle file

<a id='flask'></a>

### 2.3. Flask Web App

((t.b.d.))

<a id='run'></a>

## 3. Run the Project

Starting with the ETL process there are three steps necessary to get the WebApp in place and use the tool: 

<a id='cleaning'></a>

### 3.1. Data Cleaning

Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database:

```bat
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

The first two arguments are input data and the third argument is the SQLite Database in which we want to save the cleaned data. The ETL pipeline is in _process_data.py_.

_DisasterResponse.db_ already exists in _project's root directory_ folder but the above command will still run and replace the file with same information. 


<a id='training'></a>

### 3.2. Training ML Classifier

After the data cleaning process, run this command to run ML pipeline that trains classifier and saves ML classifier **from the project directory**:

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

This will use cleaned data to train the ML model, improve the model with grid search and saved the model to a pickle file (_classifer.pkl_).

_classifier.pkl_ already exists but the above command will still run and replace the file will new information.


<a id='runapp'></a>

### 3.3. Run the Web App

After the data cleaning process, run this command to run ML pipeline that trains classifier and saves ML classifier **from the project directory**:

```bat
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

This will use cleaned data to train the ML model, improve the model with grid search and saved the model to a pickle file (_classifer.pkl_).

_classifier.pkl_ already exists but the above command will still run and replace the file will new information.


<a id='conclusion'></a>

## 4. Conclusion

((t.b.d.))

<a id='files'></a>

## 5. File Structure

<pre>
.
├── app
│   ├── run.py------------------------# FLASK FILE THAT RUNS APP
│   ├── templates
│       ├── go.html-------------------# CLASSIFICATION RESULT PAGE
│       └── master.html---------------# MAIN PAGE OF WEB APP
├── data
│   ├── DisasterResponse.db-----------# DATABASE TO SAVE CLEANED DATA
│   ├── disaster_categories.csv-------# DATA TO PROCESS
│   ├── disaster_messages.csv---------# DATA TO PROCESS
│   └── process_data.py---------------# PERFORMS ETL PROCESS
├── images ---------------------------# PLOTS and SCREENSHOTS
├── models
│   └── classifier.pkl----------------# ML MODEL
│   └── train_classifier.py-----------# PERFORMS CLASSIFICATION TASK

</pre>


<a id='sw_requirements'></a>

## 6. Software Requirements

The project uses **Python 3.7** and additional libraries: 
- _pandas_
- _numpy_ 
- _sys_
- _time_
- _collections_
- _json_
- _re_
- _warnings_
- _operator_
- _pickle_
- _pprint_
- _flask_
- _nltk_
- _plotly_
- _scikit-learn_
- _SQLAlchemy_


<a id='links'></a>

## 7. Links


