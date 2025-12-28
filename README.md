# IIITH – Software Task 1 (SW Task-1)
### Machine Learning Regression Models for IoT Dataset
This repository contains my official **Software Task-1 assigned by IIIT Hyderabad (IIITH)**.  
The task involves applying multiple regression algorithms to an IoT dataset and comparing their performance across different environmental parameters.
This submission includes:
- Python ML code  
- Dataset  
- Performance plots  
- Final report  
- Complete documentation  

## Task Objective
The goal of this IIITH SW Task-1 is to:

- Load and understand the IoT dataset  
- Preprocess and prepare data for regression  
- Train multiple ML regression algorithms  
- Compare their output metrics  
- Visualize results through plots  
- Document the entire workflow  

## Repository Structure

SW-Task1/
│── Sameena_ACE_regression.py        # Main ML code for the task
│── Sameena_ACE_report.pdf           # Report submitted for IIITH SW Task-1
│── README.md                        # Project documentation
│
├── data/
│     ├── iot_dataset.csv
│     └── iot_dataset_mapping.csv
│
└── plots/
      ├── AQ_LinearRegression.png
      ├── AQ_RandomForestRegressor.png
      ├── AQ_XGBoostRegressor.png
      ├── SL_LinearRegression.png
      ├── SL_RandomForestRegressor.png
      ├── SL_XGBoostRegressor.png
      ├── WF_LinearRegression.png
      ├── WF_RandomForestRegressor.png
      └── WF_XGBoostRegressor.png

## Models Implemented
The following supervised regression models were implemented as per task requirements:

1. **Linear Regression**  
2. **Random Forest Regressor**  
3. **XGBoost Regressor**

Evaluation Metrics Used:
- MAE  
- MSE  
- R² Score  

## Performance Visualizations
All prediction graphs are included under the `plots/` directory, showing results for:

- **AQ** – Air Quality  
- **SL** – Solar light 
- **WF** – Water Flow  
Each parameter has 3 plots (one for each regression model).

## How to Run
### Install required libraries:
```bash
pip install -r requirements.txt
```
### Execute ML script:
```bash
python Sameena_ACE_regression.py
```
This will:
- Load the dataset  
- Train all three models  
- Print evaluation metrics  
- Generate prediction graphs
  
##  Dataset 
The dataset contains IoT readings with sensor-mapped values.  
Files included:
- `iot_dataset.csv`
- `iot_dataset_mapping.csv`
## IIITH SW Task Report
A detailed analysis with methodology, code explanation, comparisons, and conclusion is available in:
- **Sameena_ACE_report.pdf**
## 👩‍💻 Author
**Syed Sameena**  
CSE (AIDS) – B.Tech  
Software Task-1 (SW) submission  
IIIT Hyderabad – ACE
