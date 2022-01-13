# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

* This model is to predict whether one's annual salary is above or below 50k based on census data.
* Model date: Jan. 9th, 2022
* Model version: v1.0
* Model type: Random Frorest Classifier from Sci-kit learn
* Paper or other resource for more information
* License: MIT
* Contacts: ympaik@hotmail.com

## Intended Use

* This is a part of porject, Deploying a Machine Learning Model on Heroku with FastAPI from Udacity Machine Learning DevOps Engineer Couse.
* Intended to be used for being evaluated as a course work

## Training Data

* US Census Income Data Set from https://archive.ics.uci.edu/ml/datasets/census+income
* Training is done with 80% of the data.

## Evaluation Data

* Evaluation is done with 20% of the Census Income Data Set.

## Metrics

* The model is evaluated by precision, recall, and F1 scores.
* The scores are following:

    | Metrics | Scores |
    | ------- | ------ |
    | Precision | 0.8159 |
    | Recall | 0.5614 |
    | F1 | 0.6651 |

## Ethical Considerations

* The dataset contains data that could potentially discriminate against people, sensity information.

## Caveats and Recommendations

* Given gender classes are binary (male/not male), which we include as male/female. Further work needed to evaluate across a spectrum of genders.
