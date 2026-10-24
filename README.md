Early Hospital Readmission Prediction Using Explainable Ensemble Machine Learning
Project Overview
This project develops and evaluates machine-learning models to predict whether a hospital encounter involving a patient with diabetes will be followed by readmission within 30 days. It uses the Diabetes 130-US Hospitals for Years 1999–2008 dataset from the UCI Machine Learning Repository.

The project compares several classification models, selects a model using validation data, evaluates the selected model on a separate patient-grouped test set, and uses feature-importance methods and SHAP plots to help interpret model behaviour.
Important: This is an academic retrospective machine-learning project. It is not a clinically validated decision-support system and must not be used to make real patient-care decisions.

Objectives
- Explore the hospital encounter dataset and its missing values.
- Define early readmission as a binary classification target.
- Preprocess numeric and categorical features within model pipelines.
- Prevent patient overlap between training, validation, and test partitions.
- Compare baseline and ensemble classification models.
- Select a model and decision threshold using validation data only.
- Evaluate final performance on an untouched test partition.
- Inspect model behaviour using model-based feature importance, permutation importance, and SHAP explanations.
Dataset
Name: Diabetes 130-US Hospitals for Years 1999–2008
Source: UCI Machine Learning Repository

The dataset contains hospital encounters for patients with diabetes.
Dataset characteristic	Value
Encounters	101,766
Original columns	50
Unique patients	71,518
Positive target: readmitted within 30 days	11,357 (11.16%)
Negative target: not readmitted within 30 days	90,409 (88.84%)


The notebook expects the official dataset files diabetic_data.csv and IDS_mapping.csv, usually supplied in the dataset ZIP.
Target Definition
The original readmitted field is converted into a binary target:
- 1 — Early readmission: readmitted == '<30'
- 0 — Not early readmission: readmitted == 'NO' or readmitted == '>30'
The original outcome and identifier fields are not used as predictive features. encounter_id and patient_nbr are excluded from the model inputs; patient identifiers are retained only to create grouped partitions. discharge_disposition_id is also excluded under the notebook's prediction-time assumption.
Methodology
1. Data validation and exploration
The notebook loads the supplied dataset, inspects data types and missing values, checks duplicate rows and encounter IDs, and explores the distribution of the target. It does not generate synthetic replacement data if the real dataset is missing.
2. Patient-grouped data splitting
The data is split into training, validation, and test partitions while grouping by patient, so encounters belonging to the same patient do not appear in different partitions.
Partition	Encounters	Unique patients
Training	65,243	45,771
Validation	16,370	11,443
Test	20,153	14,304


The notebook reports zero patient overlap between the three partitions. The validation set is used for model selection and threshold tuning. The test set is reserved for the final evaluation.
3. Preprocessing
- Numeric missing values are imputed using the median.
- Categorical missing values are imputed using the most frequent value.
- Categorical variables are one-hot encoded.
- Numeric variables are scaled where appropriate.
- Preprocessing is fitted inside the model pipeline using training data only.
- Gradient Boosting uses a separate dense one-hot representation with infrequent categories grouped for computational feasibility.
4. Models compared
The notebook evaluates these seven candidates:
1. Dummy Classifier (prior baseline)
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. Extra Trees
6. Gradient Boosting
7. XGBoost
Compute note: Gradient Boosting uses a reproducible training-only subsample of up to 15,000 encounters and 40 boosting stages. Its training setup therefore differs from the other candidates.
5. Model selection and threshold
XGBoost is selected using validation Average Precision (PR-AUC). A classification threshold is then selected on validation predictions by maximizing F1-score, with recall used as a tie-breaker. The test set is not used to choose the model or threshold.
6. Explainability
The notebook includes:
- Model-based feature importance, where supported by the selected estimator.
- Permutation importance.
- SHAP explanations, including summary and waterfall visualizations when successfully generated.
These techniques describe patterns in the fitted model; they do not establish causation.
Results
The following results are recorded in the current notebook outputs. Re-run the notebook in your own runtime before reporting these numbers in a final submission, because results can depend on software versions and execution state.
Validation model comparison
Average Precision (PR-AUC) at the default classification threshold is used for model selection.
Model	Validation Average Precision
Dummy Classifier (prior)	0.1127
Logistic Regression	0.2082
Decision Tree	0.1875
Random Forest	0.1936
Extra Trees	0.1888
Gradient Boosting	0.1994
XGBoost	0.2136


Final test performance — XGBoost
The decision threshold was selected on the validation set and then frozen before the final test evaluation.
Metric	Test result
Accuracy	0.7559 (75.59%)
Precision	0.1902 (19.02%)
Recall	0.3952 (39.52%)
F1-score	0.2568
ROC-AUC	0.6480
Average Precision (PR-AUC)	0.1928
Frozen decision threshold	0.1349


Test confusion matrix:
	Predicted negative	Predicted positive
Actual negative	14,384	3,618
Actual positive	1,301	850


Interpretation
The target is imbalanced, with early readmission representing approximately 11% of encounters. Accuracy alone is therefore insufficient for judging performance. The selected model identifies some early-readmission cases, but precision is low, meaning many predicted positives are not positive cases under the dataset's target definition. The results should be interpreted as an academic prediction experiment, not evidence of clinical effectiveness.
Repository Contents
- HA_Final.ipynb — Jupyter/Google Colab notebook containing data exploration, preprocessing, model training, evaluation, explainability, and a demonstration prediction interface.
- README.md — Project overview, setup instructions, methodology, and results.
If your notebook has a different filename in the repository, update the filename above to match it.
Getting Started
Option A: Google Colab
1. Open the notebook in Google Colab.
2. Upload the official UCI dataset ZIP when prompted.
3. Run the notebook cells from top to bottom.
4. Review the final validation cell for dataset dimensions, split sizes, patient-overlap checks, package versions, and model metrics.
5. Confirm that all cells complete without an uncaught error before using the results in a report.
Option B: Run locally
1. Clone or download this repository.
2. Install Python and the required packages.
3. Place/extract the official dataset so the notebook can locate diabetic_data.csv and IDS_mapping.csv, or follow the notebook's upload prompt in a compatible notebook environment.
4. Open the notebook in Jupyter and run the cells in order.
Example installation command:
pip install pandas numpy scikit-learn matplotlib seaborn xgboost shap ipywidgets
The notebook records the actual runtime versions when executed. If a package compatibility error occurs, use the diagnostic output and adjust the relevant package version rather than upgrading every package indiscriminately.
Reproducibility and Design Notes
- Random seed: 42.
- Patient IDs are used for grouping only, not as model features.
- The validation set is used for model selection and threshold tuning.
- The final test set is used only for the selected model's final evaluation.
- The notebook stops if the real dataset cannot be located; it does not substitute synthetic data.
- The selected model is fitted on training plus validation data for the final evaluation/demo workflow.
- Reported metrics should be refreshed after a clean, complete notebook run.
Limitations and Responsible Use
- The dataset covers encounters from 1999–2008; patterns may not reflect current clinical practice.
- Early readmission is a minority class, and the model's precision and recall show meaningful errors remain.
- The threshold was selected to maximize validation F1 (using recall as a tie-breaker); it is not a clinically optimized threshold.
- Some features may only be available late in the hospital encounter. The notebook assumes prediction by the end of the index hospitalization and excludes discharge disposition, but the intended prediction time and availability of every feature should be reviewed carefully.
- Feature importance and SHAP values are not causal explanations.
- The model has not been prospectively or clinically validated and should not guide patient treatment or discharge decisions.
Acknowledgements
Dataset: Diabetes 130-US Hospitals for Years 1999–2008 — UCI Machine Learning Repository.
Project title: Early Hospital Readmission Prediction Using Explainable Ensemble Machine Learning
