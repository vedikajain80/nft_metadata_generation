import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.applications.imagenet_utils import decode_predictions

# Load the pre-trained model and its last convolutional layer
model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
last_conv_layer = model.get_layer('block5_conv3')

# Load the image and preprocess it
test_dir = 'test_CRYSTALS'
image_filename = "110.png"
img_path = os.path.join(test_dir, 'images', image_filename)
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.vgg16.preprocess_input(x)

# Get the output of the last convolutional layer and the predicted class index
preds = model.predict(x)
class_idx = np.argmax(preds[0])

# Decode the predicted class
pred_class = decode_predictions(preds, top=1)[0][0][1]
print(f"Predicted class: {pred_class}")

# Compute the gradient of the predicted class with respect to the output feature map of the last convolutional layer
grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
with tf.GradientTape() as tape:
    conv_output, predictions = grad_model(x)
    loss = predictions[:, class_idx]
grads = tape.gradient(loss, conv_output)[0]

# Compute the CAM
pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
grads = tf.reduce_mean(tf.multiply(conv_output, pooled_grads), axis=-1)
grads = tf.squeeze(grads)
heatmap = tf.maximum(grads, 0)
heatmap /= tf.reduce_max(heatmap)
heatmap = heatmap.numpy()

# Overlay the heatmap on the original image
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
overlayed_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
plt.imshow(overlayed_img)
plt.show()
