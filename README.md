# Data Scientist Project Project Overview
* Build a model to estimate Data Scientist Salary, may help people negotiate their wage when they get a job in Data Science
* The data available on Kaggle was scrapped from Glassdoor Website.
* By using R Program optimizes **Linear Regression**, **the Lasso**, **RandomForest** methods to reach the best model

# Code and Resource Use 

* Jupyter NoteBook **Python** ver 3.10

* **Packages**: pandas, numpy, sklearn, matplotlib, seaborn for EDA part
* **Kaggle dataset**: https://www.kaggle.com/datasets/nikhilbhathi/data-scientist-salary-us-glassdoor
* **EDA**: https://www.kaggle.com/code/nikhilbhathi/100-insights-data-science-jobs-eda/notebook
* **R Program** 

# Data Description
* Salary Estimate
* Job Title
* Rating
* Location
* Company Size
* Company Founded Year
* Type of Ownership
* Industry
* Sector
* Revenue
* Competitors
* Multiple Skills (python, spark, aws, excel etc)
* Seniority
* Degree

# EDA

I look at the distribution and values count of values. Below are some highlights:
![Data Scientist's Salary EDA - Jupyter Notebook - AVG Secure Browser 9_7_2022 11_28_28 PM](https://user-images.githubusercontent.com/99704273/188904211-e21f9aa8-e439-4fcc-9b6b-9230e872b612.png)
![image](https://user-images.githubusercontent.com/99704273/188903467-ad9b9b46-38e2-48fd-8a3c-7f9a360c5611.png)
![image](https://user-images.githubusercontent.com/99704273/188903354-7875fc28-82d3-4f5b-9243-c306cbd95831.png)
# Model Building
I transformed values of columns into readable code (filecode is attached)
Then apply different models and test the estimated the accuracy of each model based on the value Mean Squared Error (MSE).
* **Multiple Linear Regression** : the baseline model
* **The Lasso** : the techniques for shirinking the regression coefficients towards zero (Because of the sparse data from the many categorical variables, the lasso would be better)
* **Random Forest** : a tree-based method 

# Model Performance
The Random Forest model far outperformed the other approaches on the test and validation sets.

* **Random Forest** MSE = 575.75

* **The Lasso** MSE = 762.54

* **Linear Regression** MSE = 995.88

After choosing the best model to predict the salary, we can actually apply them too make predictions on future outcomes.
Maybe later on I will try to deploy the model into action !
