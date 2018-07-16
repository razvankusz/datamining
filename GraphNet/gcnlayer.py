from keras import activations, initializers, regularizers
from keras import backend as K
from keras.engine.topology import Layer


class GraphConvolution(Layer):
    def __init__(self, output_dim, kernel_regularizer=None, activation='relu', **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.kernel_regularizer = kernel_regularizer
        super(GraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dim = input_shape[0][1]

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dim, self.output_dim),
                                      initializer='uniform',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,),
                                    initializer='uniform',
                                    trainable=True)

        # Be sure to call this somewhere!
        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs):
        features = inputs[0]
        adj = inputs[1]
        output = K.dot(adj, K.dot(features, self.kernel))

        if self.bias:
            output += self.bias

        return self.activation(output)

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim)
