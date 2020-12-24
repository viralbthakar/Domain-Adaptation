from tensorflow.keras.metrics import Mean, Recall, Precision, BinaryAccuracy, Metric, CategoricalAccuracy

def get_classification_performance_metrics():
	performance_metrics = {}
	performance_metrics["train_loss"] = Mean(name="train_loss")
	performance_metrics["val_loss"] = Mean(name="val_loss")
	performance_metrics["test_loss"] = Mean(name="test_loss")
	performance_metrics["train_acc"] = CategoricalAccuracy(name="train_acc")
	performance_metrics["val_acc"] = CategoricalAccuracy(name="val_acc")
	performance_metrics["test_acc"] = CategoricalAccuracy(name="test_acc")
	return performance_metrics