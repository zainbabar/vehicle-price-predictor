# Used car price prediction

Predicting the selling price of a used car from its specs (make, year, mileage, engine, power, torque, dimensions) using the [CarDekho used car dataset](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) from Kaggle. Prices are in INR.

Everything is in [vehicle.ipynb](vehicle.ipynb) and reads top to bottom: cleaning, feature engineering, a linear regression baseline, then a few stronger models, fine tuning, and an error analysis at the end.

## Results

All of these are 10 fold cross validation on the training set. The model predicts `log(price)`, so the RMSE is converted back into real prices to keep it readable.

| Model | CV RMSE | CV R2 |
| --- | --- | --- |
| Gradient Boosting (tuned) | 574,673 | 0.958 |
| Gradient Boosting (default) | 644,277 | 0.952 |
| Random Forest | 782,801 | 0.948 |
| Linear Regression | 900,725 | 0.938 |
| Decision Tree | 1,006,883 | 0.901 |

The winner is a `GradientBoostingRegressor` with `learning_rate=0.101`, `max_depth=3` and `n_estimators=352`, found with a `RandomizedSearchCV` over 20 candidates.

On the held out test set it gets an R2 of 0.957 and an RMSE of about 1.9M. That RMSE is a lot worse than the CV number and it's basically all down to a handful of very expensive cars, see [where it falls down](#where-it-falls-down).

## The data

2,059 rows, 20 columns. I used `car-details-v4.csv` because it has the actual specs (engine, power, torque, dimensions) instead of just make/model/year.

After dropping rows with missing values there are 1,874 left, which splits into 1,499 train and 375 test. Three more rows got dropped from the training set for having impossible mileage (one car with 2 million km, one with 1 million). That leaves 1,496 training rows and 36 features after encoding.

## What I did

Feature engineering, roughly in order:

- `Make`: one hot encoded, with anything appearing fewer than 10 times bucketed into `Other`
- `Model`: dropped. There are 1,050 unique model names across 2,059 rows, so there was no sensible way to encode it for a model this simple
- `Owner`: ordinal encoded (First 1, Second 2, Third 3, UnRegistered 4)
- `Engine`, `Max Power`, `Max Torque`: parsed out of strings like `1198 cc` and `87 bhp @ 6000 rpm` into numbers, then log transformed since they were all right skewed
- `Transmission`, `Drivetrain`, `Seller Type`: one hot encoded
- `Color`, `Location`, `Fuel Type`: dropped, I didn't think they'd move the price much
- `Price`: log transformed. This was the single biggest improvement in the whole notebook. Raw prices have a long right tail, and squaring those errors in the RMSE was destroying the linear model (R2 went from 0.59 to 0.92 on the test split just from this)

Then I worked up through Linear Regression, Decision Tree, Random Forest and Gradient Boosting, evaluating each with 10 fold CV, and tuned the gradient boosting model with a randomized hyperparameter search.

## What actually drives the price

From the tuned model's feature importances:

| Feature | Importance |
| --- | --- |
| Max Power | 0.63 |
| Year | 0.12 |
| Width | 0.08 |
| Max Torque | 0.05 |
| Transmission (manual) | 0.04 |
| Kilometer | 0.03 |

Max Power dominating makes sense, expensive cars have powerful engines. I expected Make to matter more, but there were probably too many one hot columns spread too thin for the model to learn much from them, and Max Power ends up standing in for "is this a luxury car" anyway.

## Where it falls down

The model is fine on normal cars and bad on very expensive ones. The single worst test prediction is a 35M car that it underestimated by 23.5M, which by itself is about 40% of the total test RMSE. The most expensive car in the training set is 20M, so the model has never seen anything in that range and tree based models can't extrapolate past what they've seen.

Things I'd try next:

- handle the price outliers properly, or use a model family that can extrapolate (maybe blend in a linear model for the top end)
- wrap the preprocessing in a `Pipeline` / `ColumnTransformer` instead of applying the same steps twice by hand. I did the train/test split before the feature engineering, so the test set transformations are all repeated manually
- do something smarter with `Model` instead of throwing it away, since that's a lot of signal to lose

## Running it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook vehicle.ipynb
```

The csvs are already in `data/`, so the first cell (the kagglehub download) is optional.

## Layout

```
vehicle.ipynb    the whole analysis
data/            the csvs from the kaggle dataset, car-details-v4.csv is the one used
requirements.txt
```
