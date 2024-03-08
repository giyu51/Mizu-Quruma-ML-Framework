# from MizuQuruma.KNearestNeighbours import KNN

# model = KNN()

# X_train, y_train, X_test, y_test = model.generate_dataset(n_classes=2)

# model.fit(X_train, y_train)

# prediction = model.predict([[10, 10], [70, 50]])
# print(prediction)

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


from MizuQuruma.Regression import LinearRegression, StochasticLinearRegression
from icecream import ic

model = LinearRegression()
X_train, y_train = model.generate_dataset(samples=150)

model = StochasticLinearRegression()
model.fit(X_train, y_train, num_iterations=55)
