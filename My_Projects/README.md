DiaScreen AI: Clinical Diabetes Risk Screening Service

1. Problem Description
Diabetes mellitus is a silent chronic epidemic that often remains asymptomatic until severe macrovascular or microvascular complications emerge. Traditional clinical screening reliant solely on high-fasting blood glucose tests misses individuals with progressive metabolic dysfunction. 

DiaScreen AI addresses this diagnostic delay by providing a low-friction, high-sensitivity screening API powered by Gradient Boosted Decision Trees (XGBoost). By assessing multi-variable physiological indicators, biometric ratios (BMI, Waist-to-Hip ratio), lifestyle behaviors, and fasting biomarkers, the model detects subtle risk signals prior to advanced clinical manifestation.

To prioritize public health screening utility and drastically minimize false negatives (missed patients), the operational classification threshold is tuned to t = 0.2, ensuring patients at potential risk are flagged for definitive confirmatory lab evaluations.


2. Dataset
The project evaluates clinical features, risk factors, and laboratory screening parameters.
Data URL : https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset

Features included: Demographics (age, gender), Lifestyle (diet score, physical activity, alcohol/smoking profile), Biometrics (BMI, waist-to-hip ratio, blood pressure), and Glycemic markers (fasting blood glucose, HbA1c, postprandial glucose, insulin).


3. Project Structure

`notebook.ipynb`: Exploratory Data Analysis (EDA), feature importance evaluation, cross-validation, hyperparameter tuning, model selection (Decision Trees, Random Forest, XGBoost), and ROC-AUC / PR-AUC assessment.

`train.py`: Standalone script to clean data, fit the `DictVectorizer`, train the production XGBoost model, and serialize the pipeline into `DiaScreen AI.bin`.

`predict.py` / `main.py`: Production-grade REST API developed using FastAPI to serve clinical predictions and return risk levels.

`pyproject.toml` & `uv.lock`: Dependency definitions managed via Astral uv.

' Dockerfile`: Multi-stage, lean Docker container configuration for containerized deployment.