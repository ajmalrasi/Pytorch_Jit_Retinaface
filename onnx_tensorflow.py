
import onnx
from onnx_tf.backend import prepare
import numpy as np
from PIL import Image
model = onnx.load("FaceDetector.onnx") # Load the ONNX file
tf_model = prepare(model) # Import the ONNX model to Tensorflow
# img = Image.open('input.jpg').resize((224, 224))
# img = np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :]
# img = img.transpose(0,1,4,2,3)
# img = img.reshape(1,3,224,224) # Transform to Input Tensor
# Y = tf_model.run(img)[0][0] #Inference
print(tf_model)