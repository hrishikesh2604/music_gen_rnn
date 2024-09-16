# Music Generation with RNNs Using Irish Songs Dataset

## Project Overview

This project demonstrates the use of **Recurrent Neural Networks (RNNs)** for generating music based on an Irish folk music dataset. The main goal is to train an RNN model on the dataset to learn patterns in musical sequences and use the model to generate new Irish-style tunes.

This project is part of the **MIT Deep Learning Lab**, and it leverages Google Colab for the implementation. The dataset used is a collection of traditional Irish tunes encoded in ABC notation, which provides a simple text-based format for representing music.

### Link to Colab Notebook:
[Music Generation with RNNs Colab Notebook](https://colab.research.google.com/drive/1iX7Wd2272hHre4mq-eAToKGSPdQ6Xtsn?usp=sharing)

---

## Dataset

The dataset consists of **Irish folk songs** written in **ABC notation**, a compact, human-readable format for encoding music. Each song is represented as a sequence of characters, and this textual representation is used as input to the RNN for training. The RNN learns to predict the next character in a sequence based on previous characters, ultimately generating new sequences that resemble traditional Irish music.

- **ABC Notation**: A text-based music notation format that is easy to parse and manipulate. In this project, it serves as the input and output for the music generation model.

---

## Technologies Used
- **Python**: The programming language used for data processing, model creation, and training.
- **TensorFlow / Keras**: Used for building and training the RNN-based sequence model.
- **RNN (Recurrent Neural Network)**: The core of the model, designed to handle sequential data and generate new sequences.
- **Google Colab**: A cloud-based environment for running Python code with free GPU support.

---

## RNN Model Architecture

The model used in this project is an **RNN (Recurrent Neural Network)** designed to learn the structure of musical sequences. It consists of several key components:
1. **Embedding Layer**: Maps each unique character (from the ABC notation) to a dense vector representation.
2. **LSTM Layers (Long Short-Term Memory)**: A type of RNN layer designed to handle long sequences and capture dependencies between characters in the musical sequences. LSTM cells help retain memory of previous characters, making the network effective at generating coherent sequences.
3. **Dense Layer**: Outputs a probability distribution over all possible characters, from which the next character in the sequence is sampled.

---

## Key Steps in the Project

1. **Preprocessing the Dataset**: 
   - The dataset is loaded and cleaned. ABC notation is tokenized into individual characters, which are then encoded as integers for model training.
   
2. **Building the RNN Model**:
   - The RNN model, built using LSTM layers, is designed to predict the next character in a sequence. It takes sequences of a fixed length as input and outputs the probability distribution of the next character in the sequence.

3. **Training the Model**:
   - The model is trained on the dataset to minimize the loss (cross-entropy) between the predicted and actual next characters. The training process allows the RNN to learn patterns in the music.

4. **Generating New Music**:
   - Once the model is trained, it can be used to generate new music sequences. A starting sequence (seed) is provided, and the model predicts subsequent characters, forming a complete musical sequence.

---

## How to Run the Project

1. **Open the Colab Notebook**: [Music Generation with RNNs Colab Notebook](https://colab.research.google.com/drive/1iX7Wd2272hHre4mq-eAToKGSPdQ6Xtsn?usp=sharing)
2. **Upload the Dataset**: The Irish songs dataset can be uploaded and preprocessed within the notebook.
3. **Train the Model**: Follow the steps in the notebook to build and train the RNN model.
4. **Generate Music**: After training, use the model to generate new Irish-style music sequences.
5. **Visualize and Listen**: The generated sequences can be converted back into ABC notation, and you can visualize and play the music using online ABC notation converters or MIDI players.

---

## Key Concepts

- **Sequence Modeling**: The RNN learns to predict the next character in a sequence of musical notes, helping to generate new music.
- **LSTM (Long Short-Term Memory)**: A specialized RNN layer that can remember long-term dependencies in sequences, making it suitable for tasks like music generation.
- **Music Generation**: By training on a dataset of Irish songs, the RNN is able to generate music that mimics traditional Irish folk tunes.

---

## Results

After training the model, the RNN is able to generate music that resembles traditional Irish tunes. The generated sequences maintain musical structure and demonstrate the model’s ability to capture the nuances of the dataset.

---

## Future Work

- **Improving Generation Quality**: Exploring more complex RNN architectures, such as deeper LSTM networks or attention mechanisms, could improve the quality of the generated music.
- **Experimenting with Larger Datasets**: Expanding the dataset to include more songs or different genres of music could further enhance the model’s generative capabilities.
- **Interactive Music Generation**: Integrating the model into an interactive interface where users can generate music in real
