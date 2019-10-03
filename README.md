# NYC Taxi Trips

In this study we will be focused on:

 * loading a large dataset
 * creating various feature objects and injecting them into different models
 * evaluating the predictions of those models
 * utilizing tools like Deep feature synthesis and feature transformations
 * understanding metrics like feature importances, for looking at a model post-training
 
 To help us to conclude such task, we will use the package [Feature Tools](https://www.featuretools.com)
 
 # Featuretools
<a style="margin:30px" href="https://www.featuretools.com">
    <img width=50% src="https://www.featuretools.com/wp-content/uploads/2017/12/FeatureLabs-Logo-Tangerine-800.png" alt="Featuretools" />
</a>

[Featuretools](https://www.featuretools.com/) is a framework to perform automated feature engineering. It excels at transforming transactional and relational datasets into feature matrices for machine learning. This demo uses Featuretools to develop a prediction model for the New York City Taxi Trip Duration on [Kaggle](https://www.kaggle.com/c/nyc-taxi-trip-duration/overview).

Normally, solving Kaggle problems is a very iterative process. Competitors look at the dataset, determine what features they can extract, and score it with their model. They use that accuracy to make more changes to their feature extraction, and again score their model. <b>Featuretools simplifies to process to let you extract numerous features in one iteration. </b>
