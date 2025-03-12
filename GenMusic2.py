import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, Flatten, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import pretty_midi
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# Enabled mixed precision training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_bfloat16')

# Checking GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Indian Classical Music Dataset (MIDI-based)
def load_midi_files(directory):
    data = []
    if not os.path.exists(directory):
        raise ValueError(f"Dataset path '{directory}' does not exist!")

    for root, _, files in os.walk(directory):
        print(f"Scanning {root}, Found files: {files}")
        for file in files:
            if file.endswith(".midi") or file.endswith(".mid"):
                path = os.path.join(root, file)
                try:
                    print(f"Loading MIDI file: {path}")
                    midi = pretty_midi.PrettyMIDI(path)
                    notes = [note.pitch for instrument in midi.instruments for note in instrument.notes]
                    if notes:
                        data.append(notes)
                    else:
                        print(f"⚠ Warning: {file} contains no note data.")
                except Exception as e:
                    print(f" Error loading {file}: {e}")

    if len(data) == 0:
        raise ValueError("No MIDI files were successfully loaded. Check dataset path and ensure files are valid.")

    return np.array(data, dtype=object)

# Load dataset
music_data = load_midi_files("C:\\Users\\91965\\maestro-v2.0.0")

if len(music_data) == 0:
    print("No valid MIDI sequences found! Check dataset files.")
    exit()

#  Preprocess
def preprocess_midi_data(data, sequence_length=50):
    sequences, labels = [], []
    for song in data:
        song = np.array(song)
        if len(song) > sequence_length:
            for i in range(len(song) - sequence_length):
                sequences.append(song[i:i+sequence_length])
                labels.append(song[i+sequence_length])
    return np.array(sequences, dtype=np.int32), np.array(labels, dtype=np.int32)

music_data, music_labels = preprocess_midi_data(music_data, sequence_length=50)

# Music Transformer Model
vocab_size = np.max(music_data) + 1

def build_transformer_model(input_shape, head_size=64, num_heads=2, ff_dim=64, num_transformer_blocks=2):
    inputs = Input(shape=(input_shape[1],))
    x = Embedding(input_dim=vocab_size, output_dim=head_size, input_length=input_shape[1])(inputs)
    
    for _ in range(num_transformer_blocks):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(x, x)
        attn_output = Dropout(0.3)(attn_output)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        ffn_output = Dense(ff_dim, activation="relu")(x)
        ffn_output = Dropout(0.3)(ffn_output)
        ffn_output = Dense(head_size)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(vocab_size, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    return model

#  Data augmentation for 
subset_size = min(50000, len(music_data))
X_train, y_train = music_data[:subset_size], music_labels[:subset_size]

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(16).prefetch(tf.data.AUTOTUNE)

model = build_transformer_model(X_train.shape)
model.fit(train_dataset, epochs=3, validation_data=None)

# music generation
def generate_music(model, seed, num_steps=200):
    generated_sequence = list(seed)
    for _ in range(num_steps):
        input_sequence = np.array(generated_sequence[-50:]).reshape(1, -1)
        next_step = np.argmax(model.predict(input_sequence), axis=-1)[0]
        generated_sequence.append(next_step)
    return generated_sequence

#  Conversion to MIDI
def sequence_to_midi(sequence, output_file="generated_music.mid"):
    if not sequence:
        print(" Warning: Generated sequence is empty. MIDI file will not contain music.")
        return
    
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    
    for i, pitch in enumerate(sequence):
        try:
            note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=i * 0.5, end=(i+1) * 0.5)
            instrument.notes.append(note)
        except Exception as e:
            print(f"Error adding note {pitch}: {e}")
    
    if not instrument.notes:
        print("❌No valid notes were added to MIDI. Check the generated sequence.")
        return
    
    midi.instruments.append(instrument)
    midi.write(output_file)
    print(f"✅ MIDI file successfully saved as {output_file}")

#  Web App  (Streamlit)
def main():
    st.title("Indian Classical Music Generator 🎵")
    st.write("Select a Raga, Tala, and Instrument to generate music")
    
    raga = st.selectbox("Select Raga", ["Bhairav", "Yaman", "Bageshree", "Darbari", "Malkauns"])
    tala = st.selectbox("Select Tala", ["Teental", "Jhaptaal", "Ektaal", "Rupak"])
    instrument = st.selectbox("Select Instrument", ["Sitar", "Flute", "Tabla", "Violin"])
    
    if st.button("Generate Music 🎶"):
        seed = X_train[0]
        generated_sequence = generate_music(model, seed, num_steps=200)
        sequence_to_midi(generated_sequence)
        st.success("Music Generated! Download the MIDI File Below")
        st.download_button("Download MIDI", "generated_music.mid")

if __name__ == "__main__":
    main()
