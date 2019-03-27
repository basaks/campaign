# Instructions

## Working environment
Create a virtualenv with `requirements.txt`.

I used python3.6.7. 
[See instructions on how to create a python3 virtualenv](https://gist.github.com/basaks/b33ea9106c7d1d72ac3a79fdcea430eb).

Once inside your virtualenv, simply use

    pip install -r requirements.txt

## Inspirations taken from:

1. https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
2. https://www.kaggle.com/kevinarvai/fine-tuning-a-classifier-in-scikit-learn


Responses to the questions can be found in `response.ipynb`.  

## How to use
The code uses the python file `config.py` for settings directly in python.
Parameters from `config.py` are used in `campaign.py` for various jobs. The 
two main function in `campaign.py` are `config` driven and helps us 
perform various analysis. 

Since the majority and minortity classes in this dataset are imbalanced, 
there is provision in the `config.py` to use various classifiers and data 
resampling techniques from the very useful [imbalanced learn package](https://github.com/scikit-learn-contrib/imbalanced-learn). A simple class weight based approach is also attempted for each algorithm in addition to the more advanced oversampling techniques from `imblearn`.

I like to do a quick first model using a treebaed ensemble algorithm. The 
reason behind choosing a treebased classifier (RF, XGBoost) is that they 
 are very good general purpose classifiers which can be used to quickly set 
 benchmark for other, more difficult to train classifiers. 

Once a benchmark is available, based on the requirement (data volume, speed 
or training vs prediction), one can choose other classifiers. When data 
volume is low, and at the same time speed is a concern, something like 
LogisticRegression can offer good compromise.

When data volume is large, and speed of training is less important, one can 
use a stochastic gradient descent based classifier (SGDClassifier being one 
of them). Prediction of the SGDClassier is very fast.

I include a few notebooks to demonstrate how to use the functions.

## Classifiers
Most of the classifiers are taken from scikit learn, expcet from `XGBoost`. 
One could relatively easily add more classifiers. One can also 
automate/optimise the classifier and its properties, along with probably the 
class rebalancing algorithm.


## Sampling to rebalance classes
Various over/undersampling of the minority/majority class algorithms are 
available in the config file. User can simply select or automate/optimise using 
the dict keys used.


## Other parameters
Various other config parameters are available to control and run various jobs.

### GridSearch/Optimise
A grid search can be performed to optimise the hyperparameters of a 
classifier. Use `optimise=True` in the `config.py`

    optimise=True

Optimisation grid `p_grid` specific to a classifier can be supplied when 
`GridSearch` is run. Example for `LogisticRegression` and 
`RandomForestClassifier` are provided in the config file.    

The `GridSearchCV` impelemtation in the code allows optimisation based on any
 of 'precision', 'recall', 'accuracy', 'f1-score', or other custom score 
 function that can be relatively easily integrated in the setup. An example 
 of a custom custom scorer can be found in the `fbeta2` function in the 
 config file.
 
 If `optimise=True` is set in config, the `campaign.analyze` function will 
 perform hyperparamter search in in the space set by `config.p_grid`, and 
 will select the best hyperparameter combination found by grid search as  
 another potential classifier (amongst other).
    
### Feature selection

Feature selectA feature selection can be performed using 
[Recursive Feature Elimination (RFE)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
For classifiers that support a 'coef_' or 'feature_importance_'
attributes. Caution should be exercised as in the presence of 
collinear features, a process like feature importance simply replaces one 
feature with a dependent one (for example, age vs job categories in this 
dataset).

### Feature significant test
Of these selected features in the previous step, a further significance test 
is performed using `statsmodels` `Logit` class, which provides `pvalues` in 
addition to estimating the regression coefficient. Only features with 
`pvalues` less than `alpha` are accepted. Note that this step can result in 
linear algebra error if the data matrix supplied to the `Logit` class is 
singular or low rank.

### Random projections/PCA 
These were explored, but later dropped due to these feature reduction 
techniques losing the physical meaning of in the transformed space.

## General Observations and other config parameters

The dataset is imbalanced. Ratio of majority/minority class is about ~7.9. 
This requires adjustment to either the class weights, or synthetically 
oversample/undersample the minority/majority class so that the cost function 
optimised can learn both classes.

The config parameter `only_campaign=True` will analyse when `campaign=1` or 
Group B customers, who were targetted by the campaign only.

The config parameter `only_non_campaign=True` will analyse when `campaign=0` or 
(Group A - Group B) customers who were eligible for the product but not 
target by the campaign.

If both `only_non_campaign=False` and `only_campaign=False`, then the whole 
data set is analysed and the effect of the campaign can be assessed. 

## Data Prep

Since we have a mix of continuous and categorical data, we have to treat them
 accordingly. The `campaign.data_prep` class prepares the data according to 
 analysis (see config options `only_campaign` and `only_non_campaign`) being 
 performed. Using these config options we can perform a classification 
 modelling using the whole dataset (Group A), or for those that were selected
  for campaign (Group B), or those left out of the campaign (Group A- Group B).
 

### Standardise continuous features 
The continuous features are of various scales, and since certain 
classification algorithms (for example, logistic regression, svc) are scale 
dependent, we standardise the continuous variables. 

### One hot encode categorical features

The categorical variables are one hot encoded, and one one hot column is 
dropped to preserve multi-collinearity.  

### The class labels

The binary class labels are transformed using the `sklearn.LabelBinarizer`.


## Model optimisation 


### Model hyperparameter optimisation
One simple objective could be to maximise the positive class precision (TP/TP
 + FP) as this is related to a card sale, i.e, we want to correctly predict 
 the sell (TP) and also minimise the error in non-sale (FP).

At the same time, we note that we don't want to 'miss' the real sales, want 
want to minimise false negatives. So, we want to maximise the positive 
class recall (TP/(TP+FN)) as a maximal value of this quantity gives us 
minimum FN.

Alternatively, we could be optimising F1-score, which allows use to . 
Therefore, we could set `config.grid_serch_criteria=f1-score` and run our 
gridserach/hyperparameters tuning.


### Cost optimisation: A cost assumption of misclassification

A binary  classification problem is a compromise between type 1 and type 2 
error, i.e., if we are to prioritise the prediction of the positive class, the 
negative class will tend to have lower accuracy. 

In order to optimise the model, we have to assume 

    (a) cost of missing a sale due to a customer not being campaigned, and
    (b) a cost of the campaign

Let's assume (as we have not been supplied) that the cost of the campaign is 
only 10% of the cost of missing a customer who would have purchased the product
if campaigned. This will allow our classifier to hit target 
misclassifications for both 'no' (our majority class) and 'yes' (minority 
class) categories. 

This allows us to set the total number of FN and FP's for our classification.
 If we define `yes` in the response variable as our positive class (the card 
 is taken), then with the above assumption, our (cost) optimal classifier will 
 allow 10 times the number of false positives, than false negatives. This 
 also means that a higher recall is more important than precision. So while 
 picking optimisation thresholds and hyperparameters for classification, we 
 will pick one that is as low as possible in both FN and FP, with a FP ~ 10FN.
 
 To address this we can experiment with the custom `config.fbeta2` function 
 for optimisation to see if it helps speed up optimisation. 

With that out of the way, we can now choose model classifier 
threshold, hyperparameters, precision and recall. 
  

## Who we should campaign to

We should only campaign to customers that are otherwise unlikely to get the 
credit card without a campaign, and those who can be influenced to buy the 
credit card.

### Who we don't need to campaign to

There are two types of these customers from the two extremes:

(a) Those who would take the card without campaign: For this we choose a subset
 of the customers in the dataset provided who were not campaigned (Group A - 
 Group B) customers. We model these non-campaigned customers and find the  
 customers with a high `>70%` (arbitrarily set) probability of getting the card 
 without any campaign. 
 
 Since these are customers with a high probability of taking the card without a 
 campaign, we should not be wasting campaign budget on these customers.

(b) Those who were campaigned (Group B), but still came back with a low 
probability `<30%` (again arbitrarily set) of getting the card, this group 
will have even lower probability of getting the card if not campaigned. These 
are customers who would not get the card irrespective of the campaign. However, 
if this analysis is to be applied to non-campaigned customers,
we need to remove the campaign variables ("contact", "month", "day_of_week", 
"duration") from this analysis as these won't be available for non-campaigned
 customers.

These (`70%` and `30%`) probabilities will be informed by business 
requirements and can probably be optimised further based on cost of campaign 
cost vs cost due to a missed credit card sale.


### Who we need to campaign to

Customers not selected in the previous step, should be looked into further. 
These are the customers that can potentially be influenced by a 
 campaign, and campaign budget should be spent on this group. During model 
 building and data exploration we have found several factors that show the 
 effectiveness of the campaign. This group can be further filtered.
 
  
#### Explore campaign effect

I have explored the campaign effect in the `data_exploration.ipynb` notebook.
Some factors stand out from this.

We can find users with multiple attributes that should May be a recommender 
system can be designed for this purpose.
