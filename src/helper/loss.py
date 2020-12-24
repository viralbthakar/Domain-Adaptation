from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

def get_classification_loss(loss):
	if loss == 'bce':
		loss_fun = BinaryCrossentropy()
	elif loss == 'cce':
		loss_fun = CategoricalCrossentropy()
	return loss_fun

def get_optimizer(optimizer_id, learning_rate):
	if optimizer_id == 'ADAM':
		optimizer = Adam(lr = learning_rate)
	return optimizer