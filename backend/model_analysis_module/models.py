import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from model_analysis_module.utils import set_seed

def train_mlp(
    input_shape: int,
    layers: list[int],
    activation: str,
    optimizer: str,
    loss: str,
    epochs: int,
    batch_size: int,
    seed: int | None
) -> Sequential:
    """
    Builds and compiles a regression MLP with the given hyperparameters.
    """
    set_seed(seed)
    model = Sequential()
    model.add(Dense(layers[0],
                    input_shape=(input_shape,),
                    activation=activation,
                    name='Hidden_Layer_1'))
    for i, size in enumerate(layers[1:], start=2):
        model.add(Dense(size,
                        activation=activation,
                        name=f'Hidden_Layer_{i}'))
    model.add(Dense(1,
                    activation='linear',
                    name='Output_Layer'))
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['mae'])
    return model

def get_intermediate_outputs(model: Sequential, data) -> list:
    """
    Returns the output of each layer of the model for the input data `data`.
    """
    intermediate = tf.keras.Model(
        inputs=model.input,
        outputs=[layer.output for layer in model.layers]
    )
    return intermediate.predict(data)
