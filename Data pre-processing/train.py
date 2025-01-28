import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH


OUTPUT_UNITS=37
NUM_UNITS = [256]# we have a list because we may have more than one internal layer, in our case we have only one lstm layer which has 256 neurons 
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE= 0.001
EPOCHS = 50 # 48 to 100 is fine 
BATCH_SIZE = 64 # batch size is the amount of samples before running back propagation
SAVE_MODEL_PATH = "model.h5" # .h5 is the form which keras uses to save tensorflow models

def build_model(output_units, num_units, loss, learning_rate):

    # create the model architecture
    # we will use the functional api approach to build this model, we can build very complex models using functional api approach 
    input = keras.layers.Input(shape=(None, output_units))# here shape is the shape of the data that we are passing in to the network
    # here None is the first dimension, it represents the no of sequences or the number of time steps we have in the sequence that we are passing to the model 
    # when we say None here, that enables us to have as many as time steps as we want, this enables us to generate whatever length of the melodies we want 
    # the output_units, tells us the number of elements we have for each time step, turns out that we want the output size to be same as the vocabulary size 

    # now we want to add another node to our model, for doing that we should pass a new layer to the input object above
    x = keras.layers.LSTM(num_units[0])(input) # we are calling this layer on "input" layer
    x = keras.layers.Dropout(0.2)(x) # here we are adding dropout layer, to the model, dropout is a technique that is used to avoid overfitting 

    output = keras.layers.Dense(output_units, activation="softmax")(x) # here this is a softmax classifier, we have a dense layer here, the number of units in the output layer is specified and we are using softmax function as the activation function
    model = keras.Model(input, output)


    # compile the model
    model.compile(loss=loss, optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=["accuracy"]) # optimizer is the algo used to optimize and training the network, also in metrics we are just seeing the accuracy 

    model.summary()

    return model




def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # build the network
    # here we are having a custom function 

    model = build_model(output_units, num_units, loss, learning_rate) # output_units is the number of neurons in the output layer which is equal to the size of vocabulary here for this data it is 37, num_units is the number of neurons that we have in the internal layers, loss is the loss function that we will be using and learning rate goes by the definition 


    # train the model 
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model 
    model.save(SAVE_MODEL_PATH)





if __name__ == "__main__":
    train()