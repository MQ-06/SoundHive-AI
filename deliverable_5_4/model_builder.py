"""
Build sequence-based deep learning models (LSTM, GRU, RNN)
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm_model(input_shape, num_classes, lstm_units=128, dropout_rate=0.3, dense_units=64):
    """Build LSTM model for sequence classification"""
    model = Sequential()
    
    model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
    
    model.compile(
        optimizer=Adam(),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


def build_gru_model(input_shape, num_classes, gru_units=128, dropout_rate=0.3, dense_units=64):
    """Build GRU model for sequence classification"""
    model = Sequential()
    
    model.add(GRU(gru_units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
    
    model.compile(
        optimizer=Adam(),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


def build_rnn_model(input_shape, num_classes, rnn_units=128, dropout_rate=0.3, dense_units=64):
    """Build basic RNN model for sequence classification"""
    model = Sequential()
    
    model.add(SimpleRNN(rnn_units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
    
    model.compile(
        optimizer=Adam(),
        loss=loss,
        metrics=['accuracy']
    )
    
    return model


def build_model(model_type, input_shape, num_classes, units=128, dropout_rate=0.3, dense_units=64):
    """Build model based on type"""
    if model_type.upper() == "LSTM":
        return build_lstm_model(input_shape, num_classes, units, dropout_rate, dense_units)
    elif model_type.upper() == "GRU":
        return build_gru_model(input_shape, num_classes, units, dropout_rate, dense_units)
    elif model_type.upper() == "RNN":
        return build_rnn_model(input_shape, num_classes, units, dropout_rate, dense_units)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'LSTM', 'GRU', or 'RNN'")

