from keras.layers.core import Layer


class CustomZeroPadding2D(Layer):
    '''Zero-padding layer for 2D input (e.g. picture).
    # Input shape
        4D tensor with shape:
        (samples, depth, first_axis_to_pad, second_axis_to_pad)
    # Output shape
        4D tensor with shape:
        (samples, depth, first_padded_axis, second_padded_axis)
    # Arguments
        padding: tuple of int (length 4)
            How many zeros to add at the beginning and end of
            the padding dimensions (axis 3 and 4).
    '''
    input_ndim = 4

    def __init__(self, padding=(1, 1), dim_ordering='th', **kwargs):
        super(CustomZeroPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.dim_ordering == 'th':
            width = input_shape[2] + self.padding[0] + self.padding[1] if input_shape[2] is not None else None
            height = input_shape[3] + self.padding[2] + self.padding[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.dim_ordering == 'tf':
            width = input_shape[1] + self.padding[0] + self.padding[1] if input_shape[2] is not None else None
            height = input_shape[2] + self.padding[2] + self.padding[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)
