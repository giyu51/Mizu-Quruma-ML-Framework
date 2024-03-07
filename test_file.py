from MizuQuruma.KNearestNeighbours import KNN

model = KNN()

X_train, y_train, X_test, y_test = model.generate_dataset(n_classes=2)

model.fit(X_train, y_train)

prediction = model.predict([[10, 10], [70, 50]])
print(prediction)

evaluation = model.evaluate(X_test, y_test)
print(evaluation)

model.save_model()

model = model.load_model()
evaluation = model.evaluate(X_test, y_test)
print(evaluation)


my_plot = model.plot_data(X_test, y_test)
my_plot.show()

# ---

