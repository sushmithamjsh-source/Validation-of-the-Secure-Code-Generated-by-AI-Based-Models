"""
Tokenization and metric helpers used by the app.

Tokenization (getTokens / stripComments) and the F1 functions are adapted from
VUDENC (Wartschinski et al., 2022): https://github.com/LauraWartschinski/VulnerabilityDetection
"""
import tensorflow as tf
from keras import backend as K

SPLITCHARS = [" ", "\t", "\n", ".", ":", "(", ")", "[", "]", "<", ">", "+", "-", "=", "\"", "'",
              "*", "/", "\\", "~", "{", "}", "!", "?", ";", ",", "%", "&"]


def stripComments(code):
    """Remove '#' comments line by line (note: also cuts a '#' inside a string literal)."""
    lines = []
    for line in code.split("\n"):
        if "#" in line:
            line = line[:line.find("#")]
        lines.append(line)
    return "\n".join(lines) + "\n"


def getTokens(change):
    """Split code into identifier tokens and separator tokens (separators are kept)."""
    tokens = []

    change = change.replace(" .", ".")
    change = change.replace(" ,", ",")
    change = change.replace(" )", ")")
    change = change.replace(" (", "(")
    change = change.replace(" ]", "]")
    change = change.replace(" [", "[")
    change = change.replace(" {", "{")
    change = change.replace(" }", "}")
    change = change.replace(" :", ":")
    change = change.replace("- ", "-")
    change = change.replace("+ ", "+")
    change = change.replace(" =", "=")
    change = change.replace("= ", "=")
    start = 0
    for i in range(0, len(change)):
        if change[i] in SPLITCHARS:
            if i > start:
                start = start + 1
                end = i
                if start == 1:
                    token = change[:end]
                else:
                    token = change[start:end]
                if len(token) > 0:
                    tokens.append(token)
                tokens.append(change[i])
                start = i
    return tokens


# F1 loss and metric (needed as custom_objects when loading the trained models)

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        return true_positives / (predicted_positives + K.epsilon())

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))
