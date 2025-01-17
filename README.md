# S&P 500 Stock Price Prediction Using Deep Learning

This project aims to predict S&P 500 stock prices using Deep Learning models, namely LSTM, and CNN. The dataset used is historical stock price data for the S&P 500 from 2015 to 2025.

## Project Objectives
- Build Deep Learning models to predict S&P 500 stock prices.
- Compare the performance of LSTM, and CNN models.
- Provide visualizations of predicted vs actual values.

## Dataset
The dataset used is historical stock price data for the S&P 500, containing the following columns:
- `Date`: The date of the record.
- `Close/Last`: The closing price on that day.
- `Open`: The opening price on that day.
- `High`: The highest price on that day.
- `Low`: The lowest price on that day.

The dataset can be downloaded from [here](https://www.nasdaq.com/market-activity/index/spx/historical).

## Methodology
1. **Data Preprocessing**:
   - Normalize the data using `MinMaxScaler`.
   - Split the data into training set (80%) and testing set (20%).
   - Create time series dataset with `look_back` = 3.

2. **Model Building**:
   - **LSTM**: LSTM model with two LSTM layers and dropout to reduce overfitting.
   - **CNN**: CNN model with two convolutional layers and one fully connected layer.

3. **Model Evaluation**:
   - Calculate evaluation metrics such as MAE, RMSE, and R-squared.
   - Compare the performance of the three models.

4. **Visualization**:
   - Create graphs comparing predicted vs actual values for each model.

## Results
### Model Evaluation
| Model       | MAE     | RMSE    | R-squared |
|-------------|---------|---------|-----------|
| LSTM        | 0.0077  | 0.0105  | 0.7124    |
| CNN         | 0.0594  | 0.0620  | -9.0803   |

### Predicted vs Actual Graphs
![image](https://github.com/user-attachments/assets/d7ca63ad-f478-4719-a6f3-9752c049fa95)

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/nnichaelangello/Building-a-Deep-Learning-Based-Stock-Market-Prediction-System-for-the-S-P-500-Index.git
   cd repository-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook or Python script:
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   python main.py
   ```

## Dependencies
This project uses Python 3 and the following libraries:
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Contributing
If you'd like to contribute to this project, feel free to open an issue or pull request.

## License
This project is licensed under the [MIT License](LICENSE).
