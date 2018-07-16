from keras import backend as K


def apply_loss(score_array, weights):
    ''' copied from the keras library'''
    if weights is not None:
        # reduce score_array to same ndim as weight array
        ndim = K.ndim(score_array)
        weight_ndim = K.ndim(weights)
        score_array = K.mean(
            score_array, axis=list(range(weight_ndim, ndim)))
        score_array *= weights
        score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
    return K.mean(score_array)


def graph_regularisation_loss(A, weights):

    def custom_loss(y_pred, y_true):
        crossentropy = K.categorical_crossentropy(y_pred, y_true,)
        crossentropy = apply_loss(crossentropy, K.constant(weights))

        y_pred_tr = K.transpose(y_pred)
        A_ = K.constant(A)

        graph_reg = K.max(K.dot(y_pred_tr, K.dot(A_, y_pred)))

        return crossentropy + (graph_reg - 1) * 100

    return custom_loss


def crossentropy_weighted(sample_weights, gamma=1):
    def custom_loss(y_pred, y_true):
        crossentropy = K.categorical_crossentropy(y_pred, y_true)
        crossentropy = apply_loss(crossentropy, K.constant(sample_weights))
        return gamma * crossentropy
    return custom_loss


def inductive_prediction_label(y_base_cat, sample_weights):

    def custom_loss(y_pred, y_true):
        crossentropy = K.categorical_crossentropy(y_pred, y_true)
        crossentropy = apply_loss(crossentropy, K.constant(sample_weights))

        # Y = K.constant(y_base_cat)
        # crossentropy_2 = K.mean(K.categorical_crossentropy(y_pred, Y))
        # crossentropy_2 = K.print_tensor(crossentropy_2)
        # return 0 * crossentropy + crossentropy_2
        return crossentropy

    return custom_loss
