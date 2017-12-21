# Predictive-Models-for-Equipment-Fault-Detection---Secom-Dataset

secom.data: 1567 observations 590 variables (features)
secom_labels.data: classification (pass/fail) and time stamp

secom.data consists of a set of features where each data record represents a single production entity with associated measured features

secom_labels.data represent a simple pass/fail yield for in house line testing and associated data time stamp, where -1 corresponds to pass and 1 corresponds to fail and the time stamp is for that specific test point.

Various Machine Learning models are fitted to the dataset and the performances are analyzed. The model with optimal performance is chosen to predict the yield of the semiconductor manufacturing process.

Note: The data involves a special statistical scenario called Rare Events, in which, the frequency of occurence of a particular class of response variable is extremely low. Hence, sampling techniques are employed during data pre-processing stage.
