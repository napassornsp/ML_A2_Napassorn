from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt



# ============= Improt Structure of model ======================
import numpy as np
from sklearn.model_selection import KFold

class LinearRegression:
    """Custom linear regression implementation with regularization support"""
    
    kfold = KFold(n_splits=3)
    
    def __init__(self, regularization, lr=0.001, method='batch', 
                num_epochs=100, batch_size=50, cv=kfold, momentum=False):
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method = method
        self.cv = cv
        self.regularization = regularization
        self.prev_step = 0
        self.theta = None

    def r2(self, ytrue, ypred):
        y_bar = np.mean(ytrue)
        ssres = ((ypred - ytrue) ** 2).sum()
        sstot = ((ypred - y_bar) ** 2).sum()
        return 1 - (ssres/sstot)

    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]

    def xavier(self, num_features):
        lower = -1.0 / np.sqrt(num_features)
        upper = 1.0 / np.sqrt(num_features)
        return lower + (upper - lower) * np.random.rand(num_features)

    def predict(self, X):
        return X @ self.theta

    @property
    def coef_(self):
        return self.theta[1:] if self.theta is not None else None

    @property
    def intercept_(self):
        return self.theta[0] if self.theta is not None else None

# Regularization penalty classes
class NormalPenalty:
    def __init__(self, l=0.1):
        self.l = l
    def __call__(self, theta): return 0
    def derivation(self, theta): return 0

class LassoPenalty:
    def __init__(self, l=0.1):
        self.l = l
    def __call__(self, theta): return self.l * np.sum(np.abs(theta))
    def derivation(self, theta): return self.l * np.sign(theta)

class RidgePenalty:
    def __init__(self, l=0.1):
        self.l = l
    def __call__(self, theta): return self.l * np.sum(np.square(theta))
    def derivation(self, theta): return self.l * 2 * theta

class ElasticPenalty:
    def __init__(self, l=0.1, l_ratio=0.5):
        self.l = l
        self.l_ratio = l_ratio
    def __call__(self, theta): 
        return self.l * (self.l_ratio * np.sum(np.abs(theta)) + 
                       (1 - self.l_ratio) * 0.5 * np.sum(np.square(theta)))
    def derivation(self, theta):
        return self.l * (self.l_ratio * np.sign(theta) + 
                       (1 - self.l_ratio) * theta)

# Regularized regression models
class Normal(LinearRegression):
    def __init__(self, method='batch', lr=0.001, l=0.1):
        super().__init__(NormalPenalty(l), lr, method)

class Lasso(LinearRegression):
    def __init__(self, method='batch', lr=0.001, l=0.1):
        super().__init__(LassoPenalty(l), lr, method)

class Ridge(LinearRegression):
    def __init__(self, method='batch', lr=0.001, l=0.1):
        super().__init__(RidgePenalty(l), lr, method)

class ElasticNet(LinearRegression):
    def __init__(self, method='batch', lr=0.001, l=0.1, l_ratio=0.5):
        super().__init__(ElasticPenalty(l, l_ratio), lr, method)

# ===============================================================


app = Flask(__name__)

# ============= Loaded model & Scale ======================
def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Failed to load model {file_path}: {e}")
        return None


loaded_scaler = load_model('./model/pk_scale.model')
loaded_model = load_model('./model/pk_selling_price.model')
default_value = load_model('./model/pk_default_value.model')

loaded_model_v2 = load_model('./model/bestmodel.pkl')
loaded_scaler_v2 = load_model('./model/pk_scaler_poly1.pkl')


# ==============================================================


# when user open first time - Open landing page
@app.route('/')
def index():
    return render_template('home.html')




# a1 model ================================================
@app.route('/receive_data', methods=['POST'])
def receive_data():
    # Retrieve values from the form inputs; provide default values if inputs are empty
    
    engine = request.form.get('engine', default_value['engine'])  # Default to median if not provided
    power = request.form.get('max_power', default_value['max_power'])  # Ensure this matches the 'name' attribute in the HTML
    year = request.form.get('year', default_value['year'])
    km_driven = request.form.get('km_driven', default_value['km_driven'])  # Default value if not provided
    mileage = request.form.get('mileage', default_value['mileage'])  # Default value if not provided
    # Replace empty strings with default values
    if not engine:
        engine = default_value['engine']
    if not power:
        power = default_value['max_power']
    if not year:
        year = default_value['year']
    if not km_driven:
        km_driven = default_value['km_driven']
    if not mileage:
        mileage = default_value['mileage']
    print(engine, power, year, km_driven, mileage)
    try:
        # Convert the input to floats for model prediction
        input_values = pd.DataFrame([{
            'year': float(year),
            'km_driven': float(km_driven),
            'mileage': float(mileage),
            'engine': float(engine),
            'max_power': float(power)
        }])

        if loaded_scaler and loaded_model:
            scaled_input = loaded_scaler.transform(input_values)
            predicted_price = loaded_model.predict(scaled_input)
            exp_price = np.exp(predicted_price)[0]  # Assuming model outputs log prices
            print(exp_price)

            return render_template('response.html',
                                   predicted_price=f"${exp_price:.2f}")

            # return f"Predicted Price: ${exp_price:.2f}"
        else:
            raise ValueError("Model or Scaler not loaded properly.")

    except Exception as e:
        print(f"Error during request processing: {e}")
        return render_template('error.html', error=str(e))

# a2 model ================================================
@app.route('/receive_data_v2', methods=['POST'])
def receive_data_v2():
    # Retrieve values from the form inputs; provide default values if inputs are empty
    
    engine = request.form.get('engine', default_value['engine'])  # Default to median if not provided
    power = request.form.get('max_power', default_value['max_power'])  # Ensure this matches the 'name' attribute in the HTML
    year = request.form.get('year', default_value['year'])
    km_driven = request.form.get('km_driven', default_value['km_driven'])  # Default value if not provided
    mileage = request.form.get('mileage', default_value['mileage'])  # Default value if not provided
    # Replace empty strings with default values
    if not engine:
        engine = default_value['engine']
    if not power:
        power = default_value['max_power']
    if not year:
        year = default_value['year']
    if not km_driven:
        km_driven = default_value['km_driven']
    if not mileage:
        mileage = default_value['mileage']
    print(engine, power, year, km_driven, mileage)
    try:
        # Convert the input to floats for model prediction
        input_values = pd.DataFrame([{
            'year': float(year),
            'km_driven': float(km_driven),
            'mileage': float(mileage),
            'engine': float(engine),
            'max_power': float(power)
        }])

        input_values  = PolynomialFeatures(degree = 2, include_bias=False).fit_transform(input_values)

        if loaded_scaler_v2 and loaded_model_v2:
            scaled_input = loaded_scaler_v2.transform(input_values)
            intercept = np.ones((scaled_input.shape[0], 1))
            processed_input = np.concatenate((intercept, scaled_input), axis=1)
            predicted_price = loaded_model_v2.predict(processed_input)
            exp_price = np.exp(predicted_price)[0]  # Assuming model outputs log prices
            print(exp_price)

            return render_template('response.html',
                                   predicted_price=f"${exp_price:.2f}")

            # return f"Predicted Price: ${exp_price:.2f}"
        else:
            raise ValueError("Model or Scaler not loaded properly.")

    except Exception as e:
        print(f"Error during request processing: {e}")
        return render_template('error.html', error=str(e))


@app.route('/a1')
def a1():
    return render_template('a1.html')

@app.route('/a2')
def a2():
    return render_template('a2.html')





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=600)

