# test_wrapper.py
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

def angular_loss(y_true, y_pred):
    pos_true, pos_pred = y_true[:, :3], y_pred[:, :3]
    pos_loss = tf.reduce_mean(tf.square(pos_true - pos_pred))
    if y_true.shape[1] > 3:
        ang_true, ang_pred = y_true[:, 3:], y_pred[:, 3:]
        ang_diff = tf.math.floormod(ang_true - ang_pred + np.pi, 2 * np.pi) - np.pi
        ang_loss = tf.reduce_mean(tf.square(ang_diff))
        return pos_loss + 2.0 * ang_loss
    return pos_loss

model = load_model(
    '/app/output/models_flownet2pose/combined_model_final.h5',
    custom_objects={'angular_loss': angular_loss}
)

# Now run your prediction code here