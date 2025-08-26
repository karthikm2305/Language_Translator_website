# models/translator_model.py
"""
Optional Seq2Seq model wrapper.
This file is designed for experimentation. It will not run unless you call train() manually.
"""

import os
import json
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class TranslatorSeq2Seq:
    def __init__(self):
        self.encoder_model = None
        self.decoder_model = None
        self.source_tokenizer = None
        self.target_tokenizer = None
        self.max_source_len = None
        self.max_target_len = None
        self.source_vocab = None
        self.target_vocab = None
        self.latent_dim = 256
        self._loaded = False
        self._training_model = None

    def is_loaded(self):
        return self._loaded

    def train(self, source_texts, target_texts, epochs=50, batch_size=32, latent_dim=256):
        # tokenization and basic seq2seq training (adds start/end tokens if missing)
        self.latent_dim = latent_dim
        tgt = []
        for t in target_texts:
            s = t
            if not s.startswith("start "):
                s = "start " + s
            if not s.endswith(" end"):
                s = s + " end"
            tgt.append(s)

        self.source_tokenizer = Tokenizer(filters='', lower=True, oov_token='<OOV>')
        self.source_tokenizer.fit_on_texts(source_texts)
        source_seqs = self.source_tokenizer.texts_to_sequences(source_texts)
        self.max_source_len = max(len(s) for s in source_seqs)
        self.source_vocab = len(self.source_tokenizer.word_index) + 1

        self.target_tokenizer = Tokenizer(filters='', lower=True, oov_token='<OOV>')
        self.target_tokenizer.fit_on_texts(tgt)
        target_seqs = self.target_tokenizer.texts_to_sequences(tgt)
        self.max_target_len = max(len(s) for s in target_seqs)
        self.target_vocab = len(self.target_tokenizer.word_index) + 1

        encoder_input_data = pad_sequences(source_seqs, maxlen=self.max_source_len, padding='post')
        decoder_input_data = pad_sequences([seq[:-1] for seq in target_seqs], maxlen=self.max_target_len-1, padding='post')
        decoder_target_data = pad_sequences([seq[1:] for seq in target_seqs], maxlen=self.max_target_len-1, padding='post')

        # one-hot for targets
        decoder_target_onehot = np.zeros((len(decoder_target_data), self.max_target_len-1, self.target_vocab), dtype='float32')
        for i, seq in enumerate(decoder_target_data):
            for t, word in enumerate(seq):
                if word != 0:
                    decoder_target_onehot[i, t, word] = 1.0

        # encoder
        encoder_inputs = Input(shape=(self.max_source_len,), name='encoder_inputs')
        enc_emb = Embedding(input_dim=self.source_vocab, output_dim=self.latent_dim, mask_zero=True, name='enc_emb')(encoder_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_state=True, name='encoder_lstm')
        _, state_h, state_c = encoder_lstm(enc_emb)
        encoder_states = [state_h, state_c]

        # decoder
        decoder_inputs = Input(shape=(self.max_target_len-1,), name='decoder_inputs')
        dec_emb = Embedding(self.target_vocab, self.latent_dim, mask_zero=True, name='dec_emb')(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
        decoder_dense = Dense(self.target_vocab, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit([encoder_input_data, decoder_input_data], decoder_target_onehot,
                  batch_size=batch_size, epochs=epochs, validation_split=0.2)

        # build inference models
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,), name='dec_state_h')
        decoder_state_input_c = Input(shape=(self.latent_dim,), name='dec_state_c')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        dec_emb2 = model.get_layer('dec_emb')(decoder_inputs)
        decoder_lstm_layer = model.get_layer('decoder_lstm')
        decoder_outputs2, state_h2, state_c2 = decoder_lstm_layer(dec_emb2, initial_state=decoder_states_inputs)
        decoder_states2 = [state_h2, state_c2]
        decoder_dense_layer = model.get_layer('decoder_dense')
        decoder_outputs2 = decoder_dense_layer(decoder_outputs2)

        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs2] + decoder_states2)

        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self._training_model = model
        self._loaded = True

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        meta = {
            'max_source_len': self.max_source_len,
            'max_target_len': self.max_target_len,
            'source_vocab': self.source_vocab,
            'target_vocab': self.target_vocab,
            'latent_dim': self.latent_dim
        }
        with open(os.path.join(save_dir, 'meta.json'), 'w', encoding='utf8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(save_dir, 'source_tokenizer.json'), 'w', encoding='utf8') as f:
            f.write(self.source_tokenizer.to_json())
        with open(os.path.join(save_dir, 'target_tokenizer.json'), 'w', encoding='utf8') as f:
            f.write(self.target_tokenizer.to_json())
        if self._training_model is not None:
            self._training_model.save(os.path.join(save_dir, 'seq2seq_full.h5'))
        if self.encoder_model is not None:
            try:
                self.encoder_model.save(os.path.join(save_dir, 'encoder_infer.h5'))
            except Exception as e:
                print("Could not save encoder:", e)
        if self.decoder_model is not None:
            try:
                self.decoder_model.save(os.path.join(save_dir, 'decoder_infer.h5'))
            except Exception as e:
                print("Could not save decoder:", e)

    def load_model(self, save_dir):
        from tensorflow.keras.models import load_model
        with open(os.path.join(save_dir, 'meta.json'), 'r', encoding='utf8') as f:
            meta = json.load(f)
        self.max_source_len = meta['max_source_len']
        self.max_target_len = meta['max_target_len']
        self.source_vocab = meta['source_vocab']
        self.target_vocab = meta['target_vocab']
        self.latent_dim = meta.get('latent_dim', self.latent_dim)

        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        with open(os.path.join(save_dir, 'source_tokenizer.json'), 'r', encoding='utf8') as f:
            self.source_tokenizer = tokenizer_from_json(f.read())
        with open(os.path.join(save_dir, 'target_tokenizer.json'), 'r', encoding='utf8') as f:
            self.target_tokenizer = tokenizer_from_json(f.read())

        enc_path = os.path.join(save_dir, 'encoder_infer.h5')
        dec_path = os.path.join(save_dir, 'decoder_infer.h5')
        if os.path.exists(enc_path) and os.path.exists(dec_path):
            self.encoder_model = load_model(enc_path)
            self.decoder_model = load_model(dec_path)
        else:
            full = os.path.join(save_dir, 'seq2seq_full.h5')
            if os.path.exists(full):
                self._training_model = load_model(full)
                print("Full model loaded; inference models may not be available.")
            else:
                raise FileNotFoundError("No model files found in " + save_dir)
        self._loaded = True

    def translate(self, input_text, target_language='english'):
        if not self._loaded or self.encoder_model is None or self.decoder_model is None:
            raise RuntimeError("Model not loaded for inference.")

        seq = self.source_tokenizer.texts_to_sequences([input_text])
        seq = pad_sequences(seq, maxlen=self.max_source_len, padding='post')

        states_value = self.encoder_model.predict(seq)
        idx_start = self.target_tokenizer.word_index.get('start', 1)
        target_seq = np.array([[idx_start]])
        stop_condition = False
        decoded_words = []
        steps = 0
        while not stop_condition and steps < (self.max_target_len + 5):
            output_tokens_and_states = self.decoder_model.predict([target_seq] + states_value)
            output_tokens = output_tokens_and_states[0]
            h = output_tokens_and_states[1]
            c = output_tokens_and_states[2]
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.target_tokenizer.index_word.get(sampled_token_index, None)
            if sampled_word is None:
                break
            if sampled_word == 'end':
                stop_condition = True
            else:
                decoded_words.append(sampled_word)
            target_seq = np.array([[sampled_token_index]])
            states_value = [h, c]
            steps += 1
        return ' '.join(decoded_words)
