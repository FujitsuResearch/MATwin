import tensorflow as tf
import numpy as np
import sklearn.metrics as metrics

def error(actual: tf.Tensor, predicted: tf.Tensor, mask=None):
    """Calculates error from actual and predicted value and returns tensor of finite error.

    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.
        mask (tf.Tensor, optional): Boolean tensor. Defaults to None.

    Returns:
        tf.Tensor: Masked error tensor of predicted - acutual values.
    """    
    # return tf.boolean_mask(predicted - actual, tf.math.is_finite(predicted - actual))
    if mask is None:
        mask = tf.math.is_finite(predicted - actual)
    return tf.boolean_mask(predicted - actual, mask)

def l1norm(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """Calculates weighted l1 norm of finite error.

    Args:
        actual (tf.Tensor): Ground truth. 
        predicted (tf.Tensor): Prediction values.
        weight (float, optional): Weightage of l1norm metrics. Defaults to 1.

    Returns:
        tf.Tensor: Weigted l1norm of finite values.
    """    
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight * tf.norm(error(actual, predicted), 1)

def sse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """Calculates weighted sum square of finte error.

    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.
        weight (float, optional): Weightage of sse metrics. Defaults to 1.

    Returns:
        tf.Tensor: Weighted sse of finite values.
    """    
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight * tf.reduce_sum(tf.math.pow(error(actual, predicted), 2))

def mse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """Calculates weighted mean square error of finite values.

    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.
        weight (float, optional): Weightage of mse metrics. Defaults to 1.

    Returns:
        tf.Tensor: Weighted mse of finite values.
    """    
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight * tf.reduce_mean(tf.math.pow(error(actual, predicted), 2))

def mape(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """Calculates weighted mean absolute percentage error for non-zero finite observed values.
    ## Skip cases where the observed values are equal to zero or nan to avoid getting an undefined mape

    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.
        weight (float, optional): Weightage of mape metrics. Defaults to 1.

    Returns:
        tf.Tensor: Mean absolute percentage error of non-zero finite values.
    """    
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    actual, predicted = [tf.cast(tf.constant(i), tf.float32) if not isinstance(i, tf.Tensor) else tf.cast(i, tf.float32)
                         for i in [actual, predicted]]
    mask = tf.cast(tf.math.is_finite(actual), tf.int32) * tf.cast(actual > 0, tf.int32)
    # mask = tf.cast(tf.math.is_finite(actual), tf.int32)
    return 100 * weight * tf.reduce_mean(tf.abs(error(actual, predicted, mask=mask)) / tf.boolean_mask(actual, mask))

def mdape(actual: tf.Tensor, predicted: tf.Tensor):
    """Calculates median absolute percentage error for non-zero finite observed values.
    ## Skip cases where the observed values are equal to zero or nan to avoid getting an undefined mape
    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.

    Returns:
        tf.Tensor: Median absolute percentage error for non-zero finite values.
    """    
    actual, predicted = [tf.cast(tf.constant(i), tf.float32) if not isinstance(i, tf.Tensor) else tf.cast(i, tf.float32)
                         for i in [actual, predicted]]
    mask = tf.cast(tf.math.is_finite(actual), tf.int32) * tf.cast(actual > 0, tf.int32)
    # y_true, y_pred = np.array(actual), np.array(predicted)
    return 100 * np.median(tf.abs(error(actual, predicted, mask=mask)) / tf.boolean_mask(actual, mask))

def median_percentage_error(actual: tf.Tensor, predicted: tf.Tensor):
    """Calculates median percentage error for non-zero finite observed values.

    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.

    Returns:
        tf.Tensor: Median percentage error of non-zero finite values.
    """    
    mask = tf.cast(tf.math.is_finite(actual), tf.int32) * tf.cast(actual > 0, tf.int32)
    # y_true, y_pred = np.array(actual), np.array(predicted)
    return np.median(error(actual, predicted, mask=mask) / tf.boolean_mask(actual, mask)) * 100

def r2_score(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """Calculates r2_score for finite values.

    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.
        weight (float, optional): Weightage of r2_score metrics. Defaults to 1.

    Returns:
        float: r2_score value for finite values.
    """    
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    mask = tf.math.is_finite(predicted - actual)
    # if predicted.shape[0] < actual.shape[0]:
    #     predicted = tf.repeat(predicted, actual.shape[0], axis = 0)
    # print('mask', mask.shape, tf.boolean_mask(actual, mask).shape, 'actual', tf.boolean_mask(actual, mask).shape, 'predicted', tf.boolean_mask(predicted, mask).shape)
    # print("condition", tf.boolean_mask(actual, mask).shape == [0])
    if tf.boolean_mask(actual, mask).shape == [0]:
        return weight
    return weight * metrics.r2_score(y_true=tf.boolean_mask(actual, mask),
                                     y_pred=tf.boolean_mask(predicted, mask))

def rmse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """Calculated weighted root of mean square error of finite values.

    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.
        weight (float, optional): Weightage of rmse metrics. Defaults to 1.

    Returns:
        tf.Tensor: Root mean square error of finite values.
    """    
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight * tf.math.sqrt(mse(actual, predicted))

def nmse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """Calculates weighted normalized mean square error of finite values.

    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.
        weight (float, optional): Weightage of nmse metrics. Defaults to 1.

    Returns:
        tf.Tensor: Normalized mean square error of finite values.
    """    
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    # return weight*rmse(actual, predicted)/tf.experimental.numpy.nanmean(actual)
    if np.isnan(actual).all():
        return weight * mse(actual, predicted)
    return weight * mse(actual, predicted) / tf.experimental.numpy.nanmean(actual) ** 2

def nrmse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """Calculates weighted normalized root of mean square error of finite values.

    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.
        weight (float, optional): Weightage of nrmse metrics. Defaults to 1.

    Returns:
        tf.Tensor: Normalized root mean square error of finite values.
    """    
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    # return weight*rmse(actual, predicted)/tf.experimental.numpy.nanmean(actual)
    if np.isnan(actual).all():
        return weight * rmse(actual, predicted)
    return weight * rmse(actual, predicted) / tf.experimental.numpy.nanmean(actual)

def zscore(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """Calculates zscore of finite values.

    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.
        weight (float, optional): Weightage of zscore metrics. Defaults to 1.

    Returns:
        tf.Tensor: zscore of finite values.
    """    
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    # TODO: Mask standard deviation calculation
    # if np.isnan(actual).all():
    #     return weight*mse(actual, predicted)
    #
    # return weight*mse(actual, predicted)/np.nanstd(actual)**2
    if np.isnan(actual).all():
        return weight * rmse(actual, predicted)
    return weight * rmse(actual, predicted) / np.nanstd(actual)

def z2score(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """Calculates z2score of finite values.

    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.
        weight (float, optional): Weightage of z2score metrics. Defaults to 1.

    Returns:
        tf.Tensor: z2score of finite values.
    """    
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    # TODO: Mask standard deviation calculation
    if np.isnan(actual).all() or np.isnan(np.nanstd(actual)):
        return weight * mse(actual, predicted)
    return weight * mse(actual, predicted) / np.nanstd(actual) ** 2

def mnrmse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """Calculates Normalized rmse by the maximum observed value.

    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.
        weight (float, optional): Weightage of mnrmse metrics. Defaults to 1.

    Returns:
        tf.Tensor: mnrmse of finite values.
    """    
    # if weight == 0:
    #     return tf.constant(0, tf.float32)
    return weight * rmse(actual, predicted) / tf.experimental.numpy.max(actual[~tf.experimental.numpy.isnan(actual)])

def btcg_mse(actual: tf.Tensor, predicted: tf.Tensor, weight=1):
    """Normalization used by Wu et al. (2018), TRC. This metric has more numerical issues than using MSE
    
    Args:
        actual (tf.Tensor): Ground truth.
        predicted (tf.Tensor): Prediction values.
        weight (float, optional): Weightage of btcg_mse metrics. Defaults to 1.

    Returns:
        tf.Tensor: btcg_mse of finite values.
    """    
    ''' '''
    rel_error = tf.math.divide_no_nan(predicted, actual)
    return 1 / 2 * tf.reduce_mean(tf.math.pow(tf.boolean_mask(rel_error, tf.math.is_finite(rel_error)) - 1, 2))
    # return 1 / 2 * tf.reduce_mean(tf.math.pow(error(actual, predicted) /
    #                                           (tf.boolean_mask(actual, tf.math.is_finite(actual)) + epsilon), 2))
