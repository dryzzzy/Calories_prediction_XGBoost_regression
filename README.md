# XGBoost Regressor for Calorie Prediction

This project builds a robust XGBoost model to predict calories burned based on various features such as age, weight, heart rate, and body temperature.

## Key Steps:

### 1. **Data Preprocessing**
   - **Steps:**
     - Removed unnecessary columns.
     - One-hot encoded categorical variables (e.g., gender).
     - Binned the `duration` column to capture non-linear relationships and reduce noise.
     - Applied `MinMaxScaler` to normalize continuous features (`age`, `weight`, `heart_rate`, `body_temp`).
   - **Technologies Used:**
     - **`pandas`**: For data manipulation, cleaning, and one-hot encoding.
     - **`ydata_profiling`**: To generate the dataset's Profile Report for better understanding of the data distribution.
     - **`scikit-learn`**: For data preprocessing and normalization (e.g., `MinMaxScaler`).

### 2. **Model Preparation**
   - **Steps:**
     - Split data into training and test sets (70%/30%).
     - Tuned hyperparameters using `RandomizedSearchCV` to find optimal model parameters.
     - Trained the XGBoost model with the best hyperparameters.
   - **Technologies Used:**
     - **`scikit-learn`**: For train-test splitting (`train_test_split`), hyperparameter optimization (`RandomizedSearchCV`).
     - **`xgboost`**: For implementing the XGBoost Regressor model.

### 3. **Model Evaluation**
   - **Steps:**
     - Assessed model performance with metrics such as Mean Absolute Error (MAE), MSE, RMSE, R² score, and Explained Variance Score.
     - Visualized actual vs. predicted calories with a scatter plot to evaluate the model's predictive accuracy.
   - **Technologies Used:**
     - **`scikit-learn`**: For performance metrics (MAE, MSE, RMSE, R² score).
     - **`matplotlib`**: For plotting visualizations, including the scatter plot of actual vs. predicted calories.

## Conclusion
By leveraging these technologies and techniques, this project demonstrates the process of preparing a dataset, tuning an XGBoost model, and evaluating its performance for accurate calorie prediction. The XGBoost Regressor successfully handles non-linear relationships in the data and provides robust predictions with good accuracy.

## Technologies Used:
- **`pandas`**: Data manipulation and one-hot encoding.
- **`scikit-learn`**: Model evaluation, data splitting, scaling, and hyperparameter tuning.
- **`xgboost`**: Implementing the XGBoost Regressor model for prediction.
- **`matplotlib`**: For creating visualizations (scatter plots, etc.).
- **`ydata_profiling`**: For generating the Profile Report of the dataset.
