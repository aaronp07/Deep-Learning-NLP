# ANN Classification

# Simple Recurrent Neural Network
    1. Word Embedding - Text preprocessing
    2. Simple RNN
        i. Using IMDB dataset preprocessing and saved 'simple_rnn_imdb.h5'
        ii. 'simple_rnn_imdb.h5' trained data prediction done using sentiment analysis based on the review 'Positive' or 'Negative'
        iii. 'main.py' - Developed Streamlit web app


# Long Short Term Memory Recurrent Neural Network - Complete the missing word (example: I like _______ food) it will fill the data 
    1. Install nltk
    2. Download using nltk 'gutenberg'
    3. From gutenberg get the raw data of 'Shakespeare-hamlet.txt'
    4. Save into a local 'hamlet.txt' our dataset
    5. Import Data Preprocess
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from sklearn.model_selection import train_test_split
        import numpy as np
    
    6. Load the dataset
        with open('hamlet.txt', 'r') as file:
            text = file.read().lower()

    7. Tokenizer the text-creating indexes for words
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([text])
        total_words = len(tokenizer.word_index) + 1
        total_words

        # Word Index
        tokenizer.word_index
    
    8. Create Input Sequences - texts_to_sequence (Convert word into sentence line by line) 
        input_sequences = []
        for line in text.split('\n'):
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

    9. Pad Sequences - All the sentences are equal length prefix generate 0
        max_sequence_len = max([len(x) for x in input_sequences])
        # max_sequence_len
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
        input_sequences

    10. Create Predictors and Label
        import tensorflow as tf
        x, y= input_sequences[:, :-1], input_sequences[:, -1]
    
    11. Cateogorical - Wherever number see in x and y is convert into 1 rest 0
        y = tf.keras.utils.to_categorical(y, num_classes=total_words)
        y

    12. Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    13. Define early stopping
        from tensorflow.keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    14. Train our LSTM RNN
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

        # Define the model
        model = Sequential()
        model.add(Embedding(total_words, 100))
        model.add(LSTM(150, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(100))
        model.add(Dense(total_words, activation="softmax"))

        # Compile the model
        model.build(input_shape=(None, max_sequence_len))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'], callbacks=[early_stopping])
        model.summary()

    15. Function to predict the next word
        def predict_next_word(model, tokenizer, text, max_sequence_len):
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):] # Ensure the sequence length matches max_sequence_len
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        return None

    16. Input for complet the scentence and get missing word
        input_text = "That are so fortified against our"
        print(f'Input text: {input_text}')
        max_sequence_len = model.input_shape[1]+1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        print(f'Next word prediction: {next_word}')

    17. Save the model and pickle file - Why pickle file is already tokenizer generated and will use the same
        # Save the model
        model.save('next_word_lstm.h5')

        # Save the tokenizer
        import pickle
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)