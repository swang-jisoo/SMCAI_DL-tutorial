#####
# ref: https://www.tensorflow.org/tutorials/quickstart/advanced
#####

# Expert ver.
# Import libraries
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Fix the gpu memory issue
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Load mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0
#print(x_train[1])

# Add a channels dimension (adding batch)
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
#print(x_train[1])

# http://rasbt.github.io/mlxtend/user_guide/general_concepts/gradient-optimization/
# Gradient descent (GD):
# compute the cost gradient based on the full-batch (entire training dataset)
# --> slower to update te weights and longer to converge to the global cost minimum
# Stochastic gradient descent (SGD):
# update the weight after each mini-batch
# --> the path towards the global cost minimum may go zig-zag but surely faster to converge to the global cost minimum

# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#shuffle
# For SGD, the training dataset needs to be randomly mixed (shuffle) and then uniformly separated
# into mini-batchs (batch).
# shuffle(buffer_size): randomly shuffle the elements of the dataset (NOTE: training only)
# buffer_size: > sample element size
# batch(batch_size): combine consecutive elements of its input into a single batched element in the output
# batch_size: the number of consecutive elements of the dataset (normally 2^n)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(100000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# Build a subclass of tf.keras model
# Conv2D(filters, kernel_size, strides=(1,1), padding='valid'):
# capture local connectivity by sliding a feature identifier, called filter or kernel, over the image.
# filters = number of batch
# kernel_size = normally either 3 or 5
# As more convolutional layers are added, higher level of features based on the previous detected features
# will be detected.
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()

# Choose loss and optimization function for training
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

# Select metrics to accumulately store the loss and the accuracy of the model over
# epochs and print the overall result
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Use tf.GradientTape to train the model
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

# Test the model
@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

# Train and test the model
EPOCHS = 10

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )