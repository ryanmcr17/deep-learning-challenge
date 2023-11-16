# Charity Success / Funding Outcome Prediction: Deep Learning Neural Network Model Options and Recommendations

## Overview of the Analysis

* The goal of this analysis is to build a deep learning neural network model, using TensorFlow, that effectively predicts success vs. failure outcomes for individual organizations receiving funding/charity/investments. Such a model would help to minimize losses and maximize impact from the nonprofit's charity/investment decisions.
* The input/feature data used for this analysis contains key factors/features about each organization and their funding goals, including: application types, affiliations, classifications, use cases, organization types, ask/requested funding amounts, income amounts/tiers, etc.
* The goal is to predict the eventual result of each funding decision based on those factors/features, where each charity/investment decision is determined to either be successful or not. That binary outcome, i.e. "IS_SUCCESSFUL" = 0 or 1, is the label/outcome that we are trying to accurately predict.
* The input dataset on funded organizations and resulting outcomes (features + labels together) was loaded into Python dataframes, uninformative identification columns were removed, and binning was applied to variables/features with large numbers of values. The remaining feature data was then separated from the label/outcome data, the feature data was scaled, and then training and testing datasets were randomly sampled for model training+evaluation.
* An initial neural network model was fit using TensorFlow and manual selection of model details, including the number of layers, units/neurons per layer, and activation functions applied. Following that, in the separate 'AlphabetSoupCharity_Optimization.ipynb' file, auto-optimization via Keras_Tuner was applied along with additional edits/processing of the input data to create 3 additional models to attempt to reach the 75% minimum accuracy goal.

## Results

* Model 1 (initial neural network model with manually-selected parameters):
  * Accuracy: 72.75%
  * Loss: 0.5567

* Model 2 (first neural network model created via auto-optimization - initial performance was worse so edited cutoff values for binning in both the Classification and Appliation_Type columns to include more values for each):
  * Accuracy: 73.39%
  * Loss: 0.5732

* Model 3 (second neural network model created via auto-optimization - same input data as #2 except removed 'INCOME_AMT' column from input data as potentially-'confusing' feature):
  * Accuracy: 72.86%
  * Loss: 0.5651

* Model 4 (third neural network model created via auto-optimization - same input data as #3 except also removed 'STATUS' and 'SPECIAL_CONSIDERATIONS' columns as additional potentially-'confusing' features, both of which were highly imbalanced):
  * Accuracy: 73.42%
  * Loss: 0.5775

## Summary/Recommendations

* The final model created using auto-optimization and a reduced set of input variables/features showed the best performance, but the improvement in accuracy from that model was actually very minimal relative to the first manually-created model.
* With accuracy of 73.42% that final model just failed to meet the goal of >75% accuracy. All 4 of the models showed very similar accuracy, ~73% in each case.
* Still, at ~73% accuracy all these models should provide solid value in helping the nonprofit to predict likely outcomes for different charity/investment decisions they are considering, at least while additional analysis/modeling can be conducted to improve performance further.
* While the accuracy does improve very slightly with the final model, for the sake of resource/computing efficiency it would likely make sense for the organization to leverage the second model created / the first of the auto-optimized models, which produced very similar accuracy with only a single layer (again at least while continuing to iterate and improve further). However, for the sake of this challenge assignment I have saved the final model with the maximum accuracy score to the 'AlphabetSoupCharity_Optimization.h5' file.
* Given additional time/resources it would make sense to try additional changes to the input dataset to see if accuracy can be improved further. One recommendation would be to dig deeper into analysis of the distributions of values for each feature/input variable in order to identify and remove potential outlier rows/datapoints from the entire dataset. It would also make sense to bring additional data into the analysis, if available, to see if that can help improve the predictive power of the resulting models.
* If additional analysis and/or additional data is unable to improve the accuracy of models much further then it may make sense to fall back to a simpler logistic regression model for the sake of resource/computing efficiency relative to predictive accuracy.