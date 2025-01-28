import tensorflow.keras as keras
import json
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH
import numpy as np
import music21 as m21

class MelodyGenerator:

    def __init__(self, model_path="model.h5"):# this is a constructor function, we are passing arguments
        
        self.model_path= model_path # we are creating attributes for this class
        self.model= keras.models.load_model(model_path) # this loads the keras model

        # we are going to create attributes start symbols, look up table ie the mapping we made
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp) # this will help us to load the lookup table as a dictionary in our _mappings attribute

        self._start_symbols = ["/"] * SEQUENCE_LENGTH   # this is for the delimiters we have used 


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature): # seed refers to the music we input to continue the music generation based on that, num_steps is the no of steps in the time series representation that we want the network to output or generate, then max_sequence_length refers to number of steps we want to consider in the seed for the network which is equal to the sequence length here which is 64, then the final argument is temperature, temperature is a value which can value from 0 to infinity, but in our case we shall restrict it to 0 to 1, this is going to have an impact on the way we sample the output symbols
        

        # create seed with start symbol 
        seed = seed.split()
        melody = seed # melody starts with the seed
        seed = self._start_symbols + seed # we prepend start symbols to the seed, ie we will have 64 slashes 

        # map seed to integers, ie we use the lookup table and we map it to relative integers
        seed = [self._mappings[symbol] for symbol in seed] # here we are creating a list and mapping all the seed elements to the relative integers

        for _ in range(num_steps):

            # limit the seed to the max sequence length 
            seed = seed[-max_sequence_length:] # we are taking the last max sequence length
            
            # one-hot encode the seed 
            onehot_seed = keras.utils.to_categorical(seed, num_classes= len(self._mappings)) # this basically produces (max_sequence_length, no of symbols in the vocabulary) this is the dimension we get, we need a third dimesion here because we have a function 'predict' which expects a third dimension, third dimension is used to have multiple samples like we have one sample here 

            # convert and add the extra dimension ie lie (1, max_sequence_length, no of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...] # this basically adds an extra dimension 

            # make a prediction
            probabilities = self.model.predict(onehot_seed)[0] # there is a catch here, we can pass a batch of samples, so we expect a batch of probabilities back one prob distribution for each sample we pass, given we want the first and only one probability back, we index the list to zero 
            # it looks like [0.1, 0.2, 0.1, 0.6] we would have probs for each symbol probability to be the output, since we used softmax layer adding up all values will give us 1 
            # we can sample this by just taking the index of highest probability possible, but it is kind of rigid so

            output_int = self._sample_with_temperature(probabilities, temperature)

            # update the seed
            seed.append(output_int)

            # map int to our encoding 
            # that is here we map it back 
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we are at the end of a melody, that is if we get the slash symbol that means we are at the end of the melody
            if output_symbol == "/":
                break
            
            # update the melody otherwise
            melody.append(output_symbol)
        
        return melody




    def _sample_with_temperature(self, probabilities, temperature):
        # we are going to sample an index value using temperature value, 
        # if temperature goes to infinity, then the prob distri gets completely remodelled and all the different values tend to have the same distri, like a homogenous distri, so its like randomly picking one of the indexes
        # if temp goes to 0, the prob distri is going to get remodelled, and like the value which had highest probability to be picked now has probability 1 of being picked, whole thing gets highly deterministic
        # the base case is temp=1, if that is the case, then we use a normal sampling that we received out of the network 
        # the closer u get to zero, the more rigid it becomes, and the higher the value you go, more unpredictable it will be

        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions)) # we are resampling by applying softmax function

        choices = range(len(probabilities)) # this is going to look like [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilities) # to each of the choices we have a corresponding probability

        return index
    

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"): # step duration is just the amount of duration in a quarter ntoe length, in our case it is 0.25 or a sixteenth note

        # create a music21 stream 
        stream = m21.stream.Stream() # in a stream we can have subcontainers and sub sub containers, we are using the simplest default version possible of the midi stream and midi file

        # parse all the symbols in the melody and create note/rest objects
        # eg: 60 _ _ _ r _ 62 _ 
        start_symbol = None
        step_counter=1

        for i, symbol in enumerate(melody):
            # handle case in which we have a note/rest 
            if symbol != "_" or i+1 == len(melody):
                # ensure we are dealing with note/rest beyond the first one 
                if start_symbol is not None:
                    quarter_length_duration = step_duration*step_counter
                    # handle rest
                    if start_symbol == "r":
                        m21.event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21.event = m21.note.Note(int(start_symbol), quarterLength= quarter_length_duration)

                    stream.append(m21.event)

                    # reset the step counter
                    step_counter=1

                start_symbol = symbol 

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter+=1

        # write the m21 stream to a midi file
        stream.write(format, file_name)











if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"

    melody = mg.generate_melody(seed2, 500, SEQUENCE_LENGTH, 0.8)

    print(melody) 
    mg.save_melody(melody) 