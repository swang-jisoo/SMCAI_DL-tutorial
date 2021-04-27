import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initializing the input tensor
labels = ['A', 'B', 'C', 'D', 'E']
y_target = tf.constant([1,3,4,1,3,4,2,3,4,4,3,1], dtype=tf.int32)
y_pred = tf.constant([1,2,4,2,3,4,1,2,3,4,2,1], dtype=tf.int32)

# Evaluating confusion matrix
confm = np.array(tf.math.confusion_matrix(y_target, y_pred))
confm = pd.DataFrame(confm, index=labels, columns=labels)

fig = plt.figure()
ax = sns.heatmap(confm, cmap='crest', linewidths=0.5, annot=True)
# ax = sns.heatmap(confm, cmap=sns.cubehelix_palette(start=10, reverse=True), linewidths=0.5, annot=True)
plt.xlabel('True Condition')
plt.ylabel('Predicted Condition')
plt.title('Confusion Matrix')

plt.savefig('confusion_matrix.png', format='png', bbox_inches='tight')
