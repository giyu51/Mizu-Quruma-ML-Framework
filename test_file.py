# from MizuQuruma.KNearestNeighbours import KNN
# import numpy as np
# from icecream import ic


# model = KNN(regression_mode=True)

# X_train, y_train, X_test, y_test = model.generate_dataset(n_samples=200)


# model.fit(X_train, y_train)



# prediction = model.predict(X_test)
# print(prediction)
# # 
# evaluation = model.evaluate(X_test, y_test)
# print(evaluation)

# model.save_model()

# model = model.load_model()
# evaluation = model.evaluate(X_test, y_test)
# print(evaluation)


# my_plot = model.plot_data(X_test, y_test)
# my_plot.show()

# # ---

# import requests


# def get_crypto_price(symbol):
#     url = (
#         f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
#     )
#     response = requests.get(url)
#     data = response.json()
#     return data[symbol]["usd"]


# bitcoin_price = get_crypto_price("bitcoin")
# print("Bitcoin Price:", bitcoin_price)


from MizuQuruma.Regression import LinearRegression
from icecream import ic
import numpy as np


model = LinearRegression()
X_train, y_train, X_test, y_test = model.generate_dataset(
    coeff=[2], intercept=5, n_features=1, n_samples=15
)
ic(X_train)
ic(y_train)

model.fit(X_train, y_train, num_iterations=200, verbosity=1)
evaluation = model.evaluate(X_test, y_test, evaluation_metric="mse")
print(evaluation)
# ic(coeff, intercept, loss)
