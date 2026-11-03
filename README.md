# Ames Housing Price Prediction

A regression model built for a Kaggle housing price competition. The goal is to predict the sale price (`SalePrice`) of houses from 79 property features, using scikit-learn.

**Team:** Rania Aamer, Aleena Leghari, Zainab Aamer
**Course:** Data Bootcamp

## Repository Contents

- `Kaggle_Competition.ipynb` - the full walkthrough: data loading, preprocessing, model comparison, final model, and exported predictions, with commentary at each step.
- `README.md` - this file.

## Approach

1. **Data:** 1,200 training houses (79 features + `SalePrice`) and 260 test houses.
2. **Preprocessing:** missing numeric values filled with the column mean; missing categorical values filled with a "Missing" label; categorical features one-hot encoded. `MSSubClass` was treated as a category since its numbers are codes, not quantities. All steps live in a scikit-learn `Pipeline` so training and prediction are handled identically.
3. **Validation:** 80/20 train/validation split (`random_state=42`), scored with RMSE.
4. **Models compared:** a mean-price baseline, Linear Regression, Random Forest (300 trees, min leaf size 2), and Gradient Boosting.

## Results

| Model | Validation RMSE | Kaggle public RMSE |
|---|---|---|
| Linear Regression | $36,689 | 61,665 |
| Random Forest | $37,183 | 36,256 |
| **Gradient Boosting (final)** | $39,951 | **24,319** |

Our validation split favored Linear Regression, but the Kaggle public leaderboard favored Gradient Boosting by a wide margin, so we chose Gradient Boosting as our final model and retrained it on all of the training data before predicting the test set.

## How to Run

1. Download `train.csv` and `test.csv` from the Kaggle competition page.
2. Open `Kaggle_Competition.ipynb` in Google Colab (or Jupyter).
3. Run the cells in order. In Colab, the first code cell will prompt you to upload `train.csv` and `test.csv`.
4. The last cell creates `submission_gradient_boosting.csv`, which is the file submitted to Kaggle.

**Requirements:** Python 3, pandas, NumPy, scikit-learn.

## Notes

The competition data files are not included in this repository; download them from Kaggle.
