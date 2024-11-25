
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scikeras.wrappers import KerasClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, roc_auc_score, auc
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import sys
import io
import datetime
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


# sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

test_data = pd.read_csv('test_data.csv', encoding='utf-8')
train_data = pd.read_csv('train_data.csv', encoding='utf-8')

test_columns_to_keep = ['Retweet Count', 'Mention Count', 'Follower Count', 'Bot Label'] + \
                  [col for col in test_data.columns if col.startswith(('Tweet_token', 'Location_token', 'Hashtags_token'))]

train_columns_to_keep = test_columns_to_keep

X_train = train_data[train_columns_to_keep]
y_train = train_data['Bot Label']

X_test = test_data[test_columns_to_keep]
y_test = test_data['Bot Label']

print("Class distribution in traindata:")
print(train_data['Bot Label'].value_counts())

print("Class distribution in testdata:")
print(test_data['Bot Label'].value_counts())

print("Are columns consistent?")
print(set(train_columns_to_keep) == set(test_columns_to_keep))

print("Columns in train but not in test:", set(train_columns_to_keep) - set(test_columns_to_keep))
print("Columns in test but not in train:", set(test_columns_to_keep) - set(train_columns_to_keep))

X_train = X_train.iloc[-100:]
y_train = y_train.iloc[-100:]

X_test = X_test.iloc[-100:]
y_test = y_test.iloc[-100:]

y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

def add_label_noise(y, noise_level=0.1):
    noise_indices = np.random.choice(len(y), int(noise_level * len(y)), replace=False)
    y_noisy = y.copy()
    y_noisy[noise_indices] = 1 - y_noisy[noise_indices]
    return y_noisy

y_train_noisy = add_label_noise(y_train_np, noise_level=0.1)

X_train_rnn = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1)).astype(np.float32)
X_test_rnn = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1)).astype(np.float32)

X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

def create_rnn_model_with_attention(units1=64, units2=32, dropout_rate=0.5, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(X_train_rnn.shape[1], 1)))
    model.add(Bidirectional(LSTM(units1, activation='tanh', return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(units2, activation='tanh', return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Attention())
    model.add(Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter tuning for RNN with Attention
rnn_attention_param_grid = {
    'model__units1': [64],
    'model__units2': [32],
    'model__dropout_rate': [0.5],
    'model__learning_rate': [0.001],
    'batch_size': [32],
    'epochs': [50]
}

rnn_attention_model = KerasClassifier(model=create_rnn_model_with_attention, verbose=0)

rnn_attention_grid_search = GridSearchCV(estimator=rnn_attention_model, param_grid=rnn_attention_param_grid, n_jobs=-1, cv=3)
rnn_attention_grid_search.fit(X_train_rnn, y_train_noisy, validation_split=0.2,
                              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                                         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)])

best_rnn_attention = rnn_attention_grid_search.best_estimator_
best_rnn_attention_params = rnn_attention_grid_search.best_params_
rnn_attention_train_probs = best_rnn_attention.model_.predict(X_train_rnn).flatten()
rnn_attention_test_probs = best_rnn_attention.model_.predict(X_test_rnn).flatten()

# Inspect RNN with Attention Prob
print("Inspecting RNN with Attention Prob values:")
print("First 10 training RNN with Attention probabilities:")
print(rnn_attention_train_probs[:10])
print("First 10 test RNN with Attention probabilities:")
print(rnn_attention_test_probs[:10])
print("\nExplanation:")
print("- RNN with Attention Prob values represent the predicted probabilities by the RNN with Attention for each sample.")
print("- Values close to 1 indicate high confidence that the sample is a bot.")
print("- Values close to 0 indicate high confidence that the sample is not a bot.\n")

print(f"Best RNN with Attention Hyperparameters: {best_rnn_attention_params}")

# Save and reload RNN with Attention model
best_rnn_attention.model_.save('best_rnn_attention_model.keras')
best_rnn_attention.model_ = load_model('best_rnn_attention_model.keras', custom_objects={'Attention': Attention})

# Independent Evaluation: RNN with Attention
y_pred_rnn_attention = (best_rnn_attention.model_.predict(X_test_rnn).flatten() > 0.5).astype(int)

precision_rnn_attention = precision_score(y_test_np, y_pred_rnn_attention, pos_label=1, average='binary')
recall_rnn_attention = recall_score(y_test_np, y_pred_rnn_attention, pos_label=1, average='binary')
f1_rnn_attention = f1_score(y_test_np, y_pred_rnn_attention, pos_label=1, average='binary')
accuracy_rnn_attention = accuracy_score(y_test_np, y_pred_rnn_attention)

print("Independent RNN with Attention Evaluation:")
print(f"Accuracy: {accuracy_rnn_attention}")
print(f"Precision: {precision_rnn_attention}")
print(f"Recall: {recall_rnn_attention}")
print(f"F1 Score: {f1_rnn_attention}")
# Create a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Not Bot', 'Bot'],
    mode='classification'
)

# Explain a single prediction for a bot instance
bot_indices = np.where(y_test_np == 1)[0]
if len(bot_indices) > 0:
    i = bot_indices[0]  # Index of the first bot instance to explain
    exp = explainer.explain_instance(X_test.iloc[i].values, lambda x: np.hstack((1 - best_rnn_attention.model_.predict(np.reshape(x, (x.shape[0], X_test_rnn.shape[1], 1))), best_rnn_attention.model_.predict(np.reshape(x, (x.shape[0], X_test_rnn.shape[1], 1))))), num_features=59)
    exp.save_to_file('lime_explanation_bot.html')
else:
    print("No bot instances found in the test set.")

# Explanation of features that make a bot
"""
The basis of what features make a bot includes:
1. Tweet Content: The actual text of the tweet can indicate bot-like behavior, such as repetitive phrases or unnatural language patterns.
2. Retweet Count: Bots often retweet content frequently to spread information.
3. Mention Count: Bots may mention other users frequently to gain attention or spread information.
4. Follower Count: Bots may have a lower follower count compared to genuine users.
5. Verified Status: Verified accounts are less likely to be bots.
6. Temporal Features: The timing of tweets (e.g., time of day, frequency) can indicate bot-like behavior.
"""