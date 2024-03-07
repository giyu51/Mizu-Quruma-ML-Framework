import numpy as np
from typing import Iterable, Tuple
from .message_formatter import format_message


class ClassificationMetrics:
    def __init__(self, y_real: np.ndarray, y_pred: np.ndarray) -> None:
        self.y_real = self.to_numpy(y_real)
        self.y_pred = self.to_numpy(y_pred)

        self.confusion_matrix = self.calculate_confusion_matrix()
        self.tp, self.fp, self.fn, self.tn = self.calculate_confusion_metrics().values()
        self.confusion_metrics = {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }

    def calculate_accuracy(
        self, y_pred: np.array = None, y_real: np.array = None
    ) -> float:
        if y_pred is None or y_real is None:
            y_real = self.y_real
            y_pred = self.y_pred
        else:
            y_pred = self.to_numpy(y_pred)
            y_real = self.to_numpy(y_real)

        accuracy = np.mean(y_pred == y_real)

        return accuracy

    def calculate_confusion_matrix(
        self, y_real: np.ndarray = None, y_pred: np.ndarray = None
    ) -> np.ndarray:

        if y_pred is None or y_real is None:
            y_real = self.y_real
            y_pred = self.y_pred
        else:
            y_pred = self.to_numpy(y_pred)
            y_real = self.to_numpy(y_real)

        unique_labels = np.unique(np.concatenate((y_real, y_pred)))
        confusion_matrix = np.zeros(
            (len(unique_labels), len(unique_labels)), dtype=np.int32
        )

        for i, true_label in enumerate(unique_labels):
            for j, pred_label in enumerate(unique_labels):
                confusion_matrix[i, j] = np.sum(
                    (y_real == true_label) & (y_pred == pred_label)
                )

        return confusion_matrix

    def calculate_confusion_metrics(
        self,
        confusion_matrix: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if confusion_matrix is None:
            confusion_matrix = self.confusion_matrix
        else:
            confusion_matrix = self.to_numpy(confusion_matrix)

        tp = np.diag(confusion_matrix)
        fp = np.sum(confusion_matrix, axis=0) - tp
        fn = np.sum(confusion_matrix, axis=1) - tp
        tn = np.sum(confusion_matrix) - (tp + fp + fn)

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def calculate_sensitivity(
        self, tp: int | np.ndarray = None, fn: int | np.ndarray = None
    ) -> float:

        if tp is None or fn is None:
            tp = self.tp
            fn = self.fn
        else:
            tp = self.to_numpy(tp)
            fn = self.to_numpy(fn)

        if np.any(tp < 0) or np.any(fn < 0):
            message = "The number of true positives (tp) and false negatives (fn) must be non-negative."
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)

        if np.all(tp == 0) and np.all(fn == 0):
            message = "Both tp and fn cannot be zero simultaneously."
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)

        sensitivity = np.sum(tp) / (np.sum(tp) + np.sum(fn))

        return sensitivity

    def calculate_recall(
        self, tp: int | np.ndarray = None, fn: int | np.ndarray = None
    ) -> float:
        return self.calculate_sensitivity(self, tp=tp, fn=fn)

    def calculate_precision(
        self, tp: int | np.ndarray = None, fp: int | np.ndarray = None
    ) -> float:

        if tp is None or fp is None:
            tp = self.tp
            fp = self.fp
        else:
            tp = self.to_numpy(tp)
            fp = self.to_numpy(fp)

        if np.any(tp < 0) or np.any(fp < 0):
            message = "The number of true positives (tp) and false positives (fp) must be non-negative."
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)

        precision = np.sum(tp) / (np.sum(tp) + np.sum(fp))
        return precision

    def calculate_f1_score(
        self, tp: int = None, fp: int = None, fn: int = None
    ) -> float:
        if tp is None or fp is None or fn is None:
            tp = self.tp
            fp = self.fp
            fn = self.fn
        else:
            tp = self.to_numpy(tp)
            fp = self.to_numpy(fp)
            fn = self.to_numpy(fn)

        if np.any(tp < 0) or np.any(fp < 0) or np.any(fn < 0):
            message = "The number of true positives (tp), false positives (fp), and false negatives (fn) must be non-negative."
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)

        precision = self.calculate_precision(np.sum(tp), np.sum(fp))
        recall = self.calculate_sensitivity(np.sum(tp), np.sum(fn))
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    def calculate_specificity(
        self, tn: int | np.ndarray = None, fp: int | np.ndarray = None
    ) -> float:

        if tn is None or fp is None:
            tn = self.tn
            fp = self.fp
        else:
            tn = self.to_numpy(tn)
            fp = self.to_numpy(fp)

        if np.any(tn < 0) or np.any(fp < 0):

            message = "The number of true negatives (tn) and false positives (fp) must be non-negative."
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)

        specificity = np.sum(tn) / (np.sum(tn) + np.sum(fp))
        return specificity

    def to_numpy(self, _arr: Iterable) -> np.ndarray:
        try:
            numpy_array = np.array(_arr) if not isinstance(_arr, np.ndarray) else _arr
            return numpy_array

        except ValueError as ve:
            message = f"Failed to convert input to numpy array:\n {ve}"
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)

        except Exception as e:
            message = f"An unexpected error occurred:\n {e}"
            message = format_message(msg=message, msg_type="error")
            raise RuntimeError(message)

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_confusion_metrics(self):
        return self.confusion_metrics