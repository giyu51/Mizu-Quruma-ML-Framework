from typing import Literal, List, Dict, Callable, Any
import json
from icecream import ic
import numpy as np


class StrToIntEncoder:
    pass


class KNN:
    def __init__(self, user_guide=True, use_encoding=True) -> None:
        self.user_guide = user_guide
        self.user_guide_data = (
            json.loads(open("user_guide.json").read()) if self.user_guide else None
        )
        self.model_is_trained = False
        self.use_encoding = use_encoding
        self.inputs, self.outputs = self.X_train.view(), self.y_train.view()

        self.label_encoder = StrToIntEncoder()

    def _decorator_user_guide(_function: Callable[..., Any]):
        def wrapper(self, *args, **kwargs):
            if self.user_guide:
                print(_function.__name__)
                if self.user_guide_data is not None:
                    print(
                        self.user_guide_data["methods"][self.__class__.__name__][
                            _function.__name__
                        ]
                    )
                else:
                    raise FileNotFoundError(
                        "Problem reading the file `user_guide.json`"
                    )
            return _function(self, *args, **kwargs)

        return wrapper

    def _decorator_is_model_trained(_function: Callable[..., Any]):
        def wrapper(self, *args, **kwargs):
            if not self.model_is_trained:
                message = f"Before using `{_function.__name__}()` method, you must train the model. Use `fit()` method for training the model."
                raise RuntimeError(message)
            return _function(self, *args, **kwargs)

        return wrapper

    # @_decorator_is_model_trained
    @_decorator_user_guide
    def info(self):
        print("INFOMATION")

    @_decorator_user_guide
    def fit(self, X_train, y_train):

        # Update class variables accordingly to input and output arrays
        self.X_train = X_train
        self.y_train = y_train

        # Let encoder remember the whole vocabulary
        self.label_encoder.fit(vocabulary=self.y_train)

        


        self.model_is_trained = True

    def train_the_model(self, inputs, outputs):
        return self.fit(X_train=inputs, y_train=outputs)

    @_decorator_user_guide
    def evaluate(self, X_test, y_test):
        pass

    def test_the_model(self, X_test, y_test):
        return self.evaluate(X_test=X_test, y_test=y_test)

    @_decorator_user_guide
    def predict(self, X):
        predictions = [self._predict(sample) for sample in X]
        return predictions

    @_decorator_user_guide
    def _predict(self, x):
        pass

    @_decorator_user_guide
    def summary(self):
        pass

    @_decorator_user_guide
    def model_info(self):
        return self.summary()

    def generate_dataset(self):
        pass

    @_decorator_user_guide
    def plot_data(
        self, metrics=List[Literal["training", "testing", "decision_boundary"]]
    ):
        pass

    def draw_picture(
        self, metrics=List[Literal["training", "testing", "decision_boundary"]]
    ):

        return self.plot_data(metrics=metrics)


if __name__ == "__main__":
    alg = KNN(user_guide=True)
