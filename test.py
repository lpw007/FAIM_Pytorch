import tensorflow as tf


width = 128
height = 128

x_linspace = tf.linspace(-1., 1., width)
y_linspace = tf.linspace(-1., 1., height)
x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
with tf.Session() as sess:
    print(x_coordinates.eval())
    x_coordinates = tf.reshape(x_coordinates, [-1])
    print('reshape之后')
    print(x_coordinates.eval())
    y_coordinates = tf.reshape(y_coordinates, [-1])
    ones = tf.ones_like(x_coordinates)
    print('ones')
    print(ones.eval())
    indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
    print(indices_grid.eval())