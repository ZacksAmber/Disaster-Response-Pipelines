# Disaster-Response-Pipelines

## References

> [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)<br>
> 

---

## Directory and Files Explanation

### original

[original](./original.tar.gz): Original files.

---

### code

[code](./code/): Data Pipelines, Machine Learning Pipelines, NLP Pipelines, and ML models for local test.

---

### disaster_response_pipelines

[disaster_response_pipelines](./disaster_response_pipelines/): Final project contains all Python code from directory [code](./code/) and web app code.

---

#### Path inside [disaster_response_pipelines](./disaster_response_pipelines/)

```sh
disaster_response_pipelines
├── README.md
├── app
│   ├── run.py # Flask file that runs app
│   └── templates
│       ├── go.html # classification result page of web app
│       └── master.html # main page of web app
├── data
│   ├── disaster_categories.csv # data to process 
│   ├── disaster_messages.csv # data to process
│   ├── disaster_response.db # SQLite database to save clean data to. DB and TB name defined by user.
│   └── process_data.py # Data processing script. Process data from two csv files then store it into a SQLite database
└── models
    ├── model.pkl # saved model
    └── train_classifier.py
```