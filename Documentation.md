# 1. Linear Regression Module

To get started, we recommend reading up on Linear Regression, Loss, Derivatives, and Optimization.

To begin using this module, follow these steps:

1. Import the module:

   ```python
   from MizuQuruma import LinearRegression
   ```

2. Initialize the model:

   ```python
   model = LinearRegression()
   ```

3. Utilize the following methods:

   - **`fit()`** - This method fits a linear regression model to your provided data.
   - **`predict()`** - Use this method to make predictions using the trained model.
   - **`evaluate()`** - Assess the model's performance by calculating the loss on a given dataset.
   - **`plot_loss()`** - Visualize the training loss history over iterations.
   - **`save_model()`** - Save the trained model to a file with the extension `.vf`.
   - **`load_model()`** - Load a previously saved model (with the `.vf` extension) from a file.

For detailed information on each method, refer to their respective docstrings.
