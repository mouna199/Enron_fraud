# Enron_fraud

Enron group was one of the largest companies in the U.S, that collapsed in 2002 due to fraud. During the investigation, thousands of emails and financial data went public.

The goal of this project if to create a model capable of predicting people of interests based on their emails and financial information. We used 146 executives at Enron to identify the person of interest in the fraud case.

We will start by investigating and cleaning our data, then comparing two algorithms and validating the one that provided more significance and precision.

## Preprocessing:

We removed some outliers and did some feature engineering by adding two features called to_person and from_person, which represents the rate of messaged send or received from a person of interest.

## Algorithm:

We tested two algorithms, Naive bayes and Decision trees. 

Decision trees gave better results, especially when scaled using MinMaxscaler, applied PCA to select the best components and tuned with Gridsearchcv one parameter at a time.

Working with 2 componenets and 5 tree splits gave an accuracy of 0.80, a precision of 0.35 and a recall of 0.42.

## Note: Validation of the model.

A classic mistake would be to relay only on accuracy instead of precision and recall, since we have a small number of poi compared to the population size. 

Validation helps avoiding over-fitting, by splitting the data into training and testing data. we trained the model on the training data, then tested on the testing data.

Since the data is small, we made sure to split the data while keeping the same pourcentage of Poi and non Poi, for this we used StratifiedShuffleSplit, which is a technique that returns stratified randomized folds while preserving the percentage of samples for each class.
