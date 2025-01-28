import os
import music21 as m21
import json 
import tensorflow.keras as keras
import numpy as np 


KERN_DATASET_PATH = "C:\\Users\\kondr\\Desktop\\Melody_generation_MLproject\\Data pre-processing\\deutschl\\erk"
ACCEPTABLE_DURATIONS=[# These are all the durations we accept
    0.25, # we start from 0.25, the sixteenth note
    0.5, # 8th note
    0.75, # this is a dotted note, we take a 8th note and add a 16th note
    1.0, # this is the quarter note
    1.5, # dotted quarter note
    2,
    3,
    4, # this is accepting the whole note 
]
SAVE_DIR="C:\\Users\\kondr\\Desktop\\Melody_generation_MLproject\\Data pre-processing\\dataset"
SINGLE_FILE_DATASET="file_dataset"
SEQUENCE_LENGTH=64
MAPPING_PATH="mapping.json"

# kern, MIDI, MusicXML -> m21 -> kern,MIDI... 
# music21 basically allows us to use the music in object oriented way, it represents music in a object oriented way
# for eg lets say there is a composition using different instruments, here the whole composition is basically the SCORE, 
# the each instrument's role or the playing of each instrument is basically the PART, each PART has multiple MEASURES,
# MEASURES here correspond to a bar, ie to some duration all the notes, and then each bar or MEASURE has its individual NOTES



# this is to take songs with acceptable durations only
# we are doing this to simplify the note duration, to make life easier for the deep learning model 
def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:  # this flattens the whole song structure into a list of notes, a single list and when we use notes and rests it will ignore the time signature, basically filters out all the objects which are not notes or rests, so we will have only notes in the list now 
        if note.duration.quarterLength not in acceptable_durations: # .duration.quarterLength are attributes of music21
            return False
    return True


def load_songs_in_kern(dataset_path):

    songs=[]

    # go through all the files in the dataset and load them with music21
    for path,subdirs,files in os.walk(dataset_path): #this basically goes thru all the files and folder structure in the path or the parent structure recursively 
        for file in files:
            if file[-3:] == "krn":# this is used to take the consideration of the files which end with the extension krn
                song=m21.converter.parse(os.path.join(path,file)) # here path.join is going to join the path of the music file 
                # This song variable here is music21 score, or a stream, stream is the base class for objects like score and parts etc
                songs.append(song)
    return songs


# this is for transposing the notes to cmaj or amin
def transpose(song):
    # get key from the song, after analysing we found that the key is usually 
    # stored in the first measure of the first part of the score, at a specific index, which is 4 in the measure list

    parts= song.getElementsByClass(m21.stream.Part) # this will help us to get all the parts

    measures_part0= parts[0].getElementsByClass(m21.stream.Measure) # it will get the measures of the first part
    
    key= measures_part0[0][4] # in this dataset that is where the key object is stored


    # some of these songs do not have the key notated, so in that case we estimate key using music21
    if not isinstance(key,m21.key.Key): # this means the song doesn't have a key
        key = song.analyze("key") # analyze is given from music21, this basically guesses the key for the song, inbuilt function from music21
    
    # get interval for transposition, eg: Bmaj -> Cmaj, to do this we should transpose our song by an interval, so we calculate the interval
    # if it is a major it moves to Cmaj, if it is in minor it moves to Amin



    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch("C")) # it is basically calculating the interal between the key tonic and the C 
    
    elif key.mode == "minor":
         interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch("A"))

    # transpose song by the calculated interval 
    transposed_song = song.transpose(interval) # m21 gives transpose which gives us the transposed song 

    return transposed_song

    # why are we transposing all to Cmaj or Amin, because we dont want to learn all the keys, rather it will just learn cmaj and amin, so that it will use less data rather than generalising it to 24 keys 


def encode_song(song, time_step=0.25):# time_step is amount of time for each step
    # lets say we have pitch=60, duration=1.0 --> this shall be encoded as [60,"_","_","_"]
    
    encoded_song=[]
    
    for event in song.flat.notesAndRests:
        # to handle notes
        if isinstance(event,m21.note.Note):
            symbol=event.pitch.midi # here in example it is equal to 60
        
        elif isinstance(event,m21.note.Rest):
            symbol = "r"
        
        # now we need to convert note/rest into time series notation
        steps = int(event.duration.quarterLength/time_step)# no of steps required for the current symbol or event to be represented in time series form 

        for step in range(steps):
            if step == 0:# only the first one will be the value and we hold the event for whatever duration is left, other than that all will be the holding the note if it is continued, the first one will be either the value or the rest value, after that it if the note is held we will have the underscore
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # we want to cast the encoded song to a string 
    encoded_song=" ".join(map(str,encoded_song)) # we are mapping all the items to strings, ie casting all the items to strings

    return encoded_song








def preprocess(dataset_path):


    # load the folk songs
    print("Loading songs...")
    songs=load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs")

    for i,song in enumerate(songs):

        # filter out songs that have non acceptable durations
        if not has_acceptable_durations(song,ACCEPTABLE_DURATIONS):
            continue 


        # transpose songs to Cmaj or Amin
        song= transpose(song)

        # encode songs with music time series representation
        encoded_song=encode_song(song)

        # save songs to text file
        save_path=os.path.join(SAVE_DIR, str(i))
        with open(save_path,"w") as fp:
            fp.write(encoded_song)


def load(file_path):
    with open(file_path,"r") as fp:
        song = fp.read()
    return song





def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length): # here dataset_path is the path to the dataset folder

    # this is used to understand that a particular song has ended and a new one has started, these are called delimiters
    new_song_delimiter = "/ " * sequence_length # when we are training our lstm we need to pass some sequences which have a fixed length, here we want to use 64 items, that is we want to use the same amount of slash symbols for a delimiter, so that at the end of a song we are going to have an amount of delimiters that are same as the no of items we have in a sequence 
    songs = ""
    
    # load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path= os.path.join(path, file)
            song = load(file_path) # this is just going to be a string like we have seen before for each song 
            songs = songs + song + " " + new_song_delimiter
    songs = songs[:-1] # this is to remove the last space after slash symbol

    # save the string that contains all data
    with open(file_dataset_path,"w") as fp:
        fp.write(songs)
    
    return songs



# so here one more problem is the neural networks we work with cannot read strings, they only read integers, so we need to map the symbols to integers, making a look up table sort of a thing 
def create_mapping(songs, mapping_path):
    mappings={}
    # identify the vocabulary
    songs =  songs.split() # this is used to split the string into its components, it is a list with all the symbols in the dataset
    vocabulary = list(set(songs)) # we are basically casting the songs list to a set, and then going back to a list


    # create mappings
    for i,symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save the vocabulary to a json file
    with open(mapping_path,"w") as fp:
        json.dump(mappings,fp, indent=4)


def convert_songs_to_int(songs):# this will basically convert the string into integer format
    int_songs = []

    # load the mappings
    with open(MAPPING_PATH,"r") as fp:
        mappings = json.load(fp)

    # cast songs string to a list
    songs = songs.split()

    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs 


# inorder to train our neural network we are going to get our sequences, that basically are the subsets of our time series or song dataset represented with time series,
#  we are going to feed these sequences into the network and then we are going to have targets for that as well, this is a supervised problem, so we have inputs as well as relative targets, inputs are fixed length sequences, targets are the values that come after each sequence 
def generate_training_sequences(sequence_length):
    # [11, 12, 13, 14 ....] -> i: [11, 12], t: 13; i: [12, 13], t: 14.... we can continue like this until we hit the end of the time series 
    # we are passing in inputs and asking the network to predict the next element in the time series
    

    # load the songs and map them to int
    songs= load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    # generate the training sequences
    # lets say we have a dateset which has 100 symbols, 64 sequence length, how many sequences can be generated? -> we can generate 100-64 = 36 sequences, because each sequence has 64 length

    inputs=[]
    targets=[]

    num_sequences = len(int_songs) - sequence_length

    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length]) # at each step takes slice of the int_songs of the time series and when we increase the counter we simply slide to the right by one step 
        targets.append(int_songs[i+sequence_length])


    # one-hot encode the sequences 
    # lets look at the shape of inputs, it is 2 dimensional list
    # inputs: (no of sequences, sequence length)
    # lets take an example : lets say the following are the different sequences, categorical data, lets say here we have vocabulary size is equal to 3 
    # [ [0, 1, 2] [1, 1, 2] ] -> [ [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ], [ [], [], [] ]]
    # one hot encoding uses no of items which is equal to vocabulary size of the dataset, here example is 3
    # each position here represents a class
    # the number of units in input layer in the network is going to be equal to the vocabulary size of our dataset, number of items in each of the arrays in one hot encoding  
    # we do one hot encoding because this is the easiest way to deal with categorical data with neural networks
    # here in the json file, we have 18 as the vocab size, so we have size as 18
    # so input will now transform to : inputs: (no of sequences, sequence length, vocabulary size)

    vocabulary_size= len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size) # this is one hot encoding, feature provided by keras
    targets= np.array(targets) # we just converted the list into an array

    return inputs, targets







def main():
    preprocess(KERN_DATASET_PATH)
    songs= create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH) 
    create_mapping(songs,MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    
    




if __name__ == "__main__":
    main()
    


