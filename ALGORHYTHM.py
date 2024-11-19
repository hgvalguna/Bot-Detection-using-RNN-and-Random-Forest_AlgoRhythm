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

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

test_data = pd.read_csv('D:/Games/School Stuff/Github/THESIS2_ALGORHYTHM/test_data.csv', encoding='utf-8')
train_data = pd.read_csv('D:/Games/School Stuff/Github/THESIS2_ALGORHYTHM/train_data.csv', encoding='utf-8')

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

X_train = X_train.iloc[-500:]
y_train = y_train.iloc[-500:]

X_test = X_test.iloc[-500:]
y_test = y_test.iloc[-500:]

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

# Create RNN model
def create_rnn_model(units1=64, units2=32, dropout_rate=0.5, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(X_train_rnn.shape[1], 1)))
    model.add(Bidirectional(LSTM(units1, activation='tanh', return_sequences=True)))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(units2, activation='tanh')))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter tuning for RNN
rnn_param_grid = {
    'model__units1': [64, 128],
    'model__units2': [32, 64],
    'model__dropout_rate': [0.3, 0.5, 0.7],
    'model__learning_rate': [0.001, 0.0005],
    'batch_size': [32, 64],
    'epochs': [50, 100]
}

rnn_model = KerasClassifier(model=create_rnn_model, verbose=0)

rnn_grid_search = GridSearchCV(estimator=rnn_model, param_grid=rnn_param_grid, n_jobs=-1, cv=3)
rnn_grid_search.fit(X_train_rnn, y_train_noisy, validation_split=0.2,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)])

best_rnn = rnn_grid_search.best_estimator_
best_rnn_params = rnn_grid_search.best_params_
rnn_train_probs = best_rnn.model_.predict(X_train_rnn).flatten()
rnn_test_probs = best_rnn.model_.predict(X_test_rnn).flatten()

# Inspect RNN_Prob
print("Inspecting RNN_Prob values:")
print("First 10 training RNN probabilities:")
print(rnn_train_probs[:10])
print("First 10 test RNN probabilities:")
print(rnn_test_probs[:10])
print("\nExplanation:")
print("- RNN_Prob values represent the predicted probabilities by the RNN for each sample.")
print("- Values close to 1 indicate high confidence that the sample is a bot.")
print("- Values close to 0 indicate high confidence that the sample is not a bot.\n")

print(f"Best RNN Hyperparameters: {best_rnn_params}")

# Save and reload RNN model
best_rnn.model_.save('D:/Games/School Stuff/Github/THESIS2_ALGORHYTHM/best_rnn_model.keras')
best_rnn.model_ = load_model('D:/Games/School Stuff/Github/THESIS2_ALGORHYTHM/best_rnn_model.keras')

# Independent Evaluation: RNN
y_pred_rnn = (best_rnn.model_.predict(X_test_rnn).flatten() > 0.5).astype(int)

precision_rnn = precision_score(y_test_np, y_pred_rnn, pos_label=1, average='binary')
recall_rnn = recall_score(y_test_np, y_pred_rnn, pos_label=1, average='binary')
f1_rnn = f1_score(y_test_np, y_pred_rnn, pos_label=1, average='binary')
accuracy_rnn = accuracy_score(y_test_np, y_pred_rnn)

print("Independent RNN Evaluation:")
print(f"Accuracy: {accuracy_rnn}")
print(f"Precision: {precision_rnn}")
print(f"Recall: {recall_rnn}")
print(f"F1 Score: {f1_rnn}")

# Perform cross-validation
cv_scores_rnn = cross_val_score(
    rnn_model,                # Your model
    X_train_rnn,              # Training features
    y_train_noisy,            # Training labels
    cv=5,                     # Number of cross-validation folds
    scoring='accuracy',       # Evaluation metric
    n_jobs=-1                 # Parallel processing
)

# Output cross-validation scores and mean accuracy
print("Cross-validation Scores (RNN):", cv_scores_rnn)
print("Mean CV Accuracy (RNN):", cv_scores_rnn.mean())

# Independent Evaluation: Random Forest
rf_independent = RandomForestClassifier(random_state=42)
rf_param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf_grid_search_independent = GridSearchCV(
    estimator=rf_independent,
    param_grid=rf_param_grid,
    cv=5,
    n_jobs=-1,
    scoring='f1'
)
rf_grid_search_independent.fit(X_train, y_train_noisy)

# Get the best hyperparameters
best_rf_params = rf_grid_search_independent.best_params_
print("Best Random Forest Hyperparameters:")
print(best_rf_params)

# Use the best model to predict
y_pred_rf_independent = rf_grid_search_independent.best_estimator_.predict(X_test)

# Evaluate the best model
precision_rf_independent = precision_score(y_test, y_pred_rf_independent, pos_label=1, average='binary')
recall_rf_independent = recall_score(y_test, y_pred_rf_independent, pos_label=1, average='binary')
f1_rf_independent = f1_score(y_test, y_pred_rf_independent, pos_label=1, average='binary')
accuracy_rf_independent = accuracy_score(y_test, y_pred_rf_independent)

print("Independent RF Evaluation:")
print(f"Accuracy: {accuracy_rf_independent}")
print(f"Precision: {precision_rf_independent}")
print(f"Recall: {recall_rf_independent}")
print(f"F1 Score: {f1_rf_independent}")

cv_scores_rf = cross_val_score(
    rf_independent,        # Random Forest model
    X_train,               # Training features
    y_train_noisy,         # Training labels
    cv=5,                  # Number of cross-validation folds
    scoring='f1',          # Evaluation metric (F1-score)
    n_jobs=-1              # Parallel processing
)

# Output cross-validation scores and mean F1 score
print("Cross-validation Scores (Random Forest):", cv_scores_rf)
print("Mean CV F1 Score (Random Forest):", cv_scores_rf.mean())

# Stacked Model Evaluation
rnn_train_probs = best_rnn.model_.predict(X_train_rnn).flatten()
rnn_test_probs = best_rnn.model_.predict(X_test_rnn).flatten()

X_train_rf = X_train.copy()
X_train_rf['RNN_Prob'] = rnn_train_probs

X_test_rf = X_test.copy()
X_test_rf['RNN_Prob'] = rnn_test_probs

# Random Forest with RNN probability as an additional feature

stacked_param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

stacked_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=stacked_param_grid,
    cv=5,
    n_jobs=-1,
    scoring='f1'
)

stacked_grid_search.fit(X_train_rf, y_train_noisy)

# Best Random Forest model after RNN stacking
best_stacked_model = stacked_grid_search.best_estimator_

# Retrieve best hyperparameters
best_rf_params = stacked_grid_search.best_params_
print("Best Stacked Model Hyperparameters:")
print(best_rf_params)

# Feature importances and feature names
feature_importances = best_stacked_model.feature_importances_
feature_names = X_train_rf.columns

y_pred_rf = best_stacked_model.predict(X_test_rf)

# Evaluation of the stacked model
precision_stacked = precision_score(y_test, y_pred_rf, pos_label=1, average='binary')
recall_stacked = recall_score(y_test, y_pred_rf, pos_label=1, average='binary')
f1_stacked = f1_score(y_test, y_pred_rf, pos_label=1, average='binary')
accuracy_stacked = accuracy_score(y_test, y_pred_rf)

print("Stacked Model Evaluation (RF with RNN Probabilities):")
print(f"Accuracy: {accuracy_stacked}")
print(f"Precision: {precision_stacked}")
print(f"Recall: {recall_stacked}")
print(f"F1 Score: {f1_stacked}")

# Perform cross-validation for the stacked model
cv_scores_stacked = cross_val_score(
    RandomForestClassifier(random_state=42),  # Random Forest model
    X_train_rf,                               # Training features (includes RNN probabilities)
    y_train_noisy,                            # Training labels
    cv=5,                                     # Number of cross-validation folds
    scoring='f1',                             # Evaluation metric (F1-score)
    n_jobs=-1                                 # Parallel processing
)

# Output cross-validation scores and mean F1 score
print("Cross-validation Scores (F1) for Stacked Model:", cv_scores_stacked)
print("Mean CV F1 Score for Stacked Model:", cv_scores_stacked.mean())

plt.figure(figsize=(12, 8))
plt.title("Feature Importances", fontsize=16)

sorted_indices = np.argsort(feature_importances)[::-1]

plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
plt.xticks(range(len(feature_importances)), feature_names[sorted_indices], rotation=90)
plt.xlim([-1, len(feature_importances)])
plt.tight_layout()
plt.show()

mda_results = permutation_importance(best_stacked_model, X_test_rf, y_test, n_repeats=30, random_state=42)
sorted_idx = mda_results.importances_mean.argsort()[::-1]

plt.figure(figsize=(12, 8))
plt.title("Permutation Importance (Mean Decrease Accuracy)", fontsize=16)

plt.barh(range(len(sorted_idx)), mda_results.importances_mean[sorted_idx], xerr=mda_results.importances_std[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
plt.tight_layout()
plt.show()

# RECURRENT NEURAL NETWORK ROC AUC CURVE
rnn_train_auc = roc_auc_score(y_train, rnn_train_probs)
rnn_test_auc = roc_auc_score(y_test, rnn_test_probs)

fpr_rnn, tpr_rnn, _ = roc_curve(y_test, rnn_test_probs)
roc_auc_rnn = auc(fpr_rnn, tpr_rnn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rnn, tpr_rnn, label=f'RNN AUC = {roc_auc_rnn:.3f}', color='blue')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - RNN')
plt.legend(loc='lower right')
plt.show()

# RANDOM FOREST ROC AUC CURVE
rf_test_probs = best_stacked_model.predict_proba(X_test_rf)[:, 1]
rf_auc = roc_auc_score(y_test, rf_test_probs)

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_test_probs)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest AUC = {roc_auc_rf:.3f}', color='green')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()

# STACKED MODEL ROC AUC CURVE
stacked_test_probs = (rnn_test_probs + rf_test_probs) / 2
stacked_auc = roc_auc_score(y_test, stacked_test_probs)
fpr_stacked, tpr_stacked, _ = roc_curve(y_test, stacked_test_probs)
roc_auc_stacked = auc(fpr_stacked, tpr_stacked)

plt.figure(figsize=(8, 6))
plt.plot(fpr_stacked, tpr_stacked, label=f'Stacked Model AUC = {roc_auc_stacked:.3f}', color='red')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Stacked Model')
plt.legend(loc='lower right')
plt.show()

# AUC VALUES
print(f"RNN Test AUC: {rnn_test_auc:.3f}")
print(f"Random Forest Test AUC: {rf_auc:.3f}")
print(f"Stacked Model Test AUC: {stacked_auc:.3f}")

# Ensure the data is properly preprocessed for CNN
X_train_cnn = X_train_rnn  # 3D array already reshaped for RNNs (samples, timesteps, features)
X_test_cnn = X_test_rnn  # Reuse reshaped arrays

# Define the 1D-CNN model
def create_1d_cnn(input_shape, filters=64, kernel_size=3, dense_units=64, dropout_rate=0.5):
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),

        Conv1D(filters=filters * 2, kernel_size=kernel_size, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        
        Dense(dense_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
cnn_model = create_1d_cnn(input_shape=(X_train_cnn.shape[1], X_train_cnn.shape[2]))
cnn_history = cnn_model.fit(X_train_cnn, y_train_noisy, epochs=50, batch_size=32, validation_split=0.2,
                            callbacks=[
                                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
                            ])

# Evaluate on test data
cnn_test_probs = cnn_model.predict(X_test_cnn).flatten()
cnn_test_preds = (cnn_test_probs >= 0.5).astype(int)

# Metrics Calculation
cnn_accuracy = accuracy_score(y_test, cnn_test_preds)
cnn_precision = precision_score(y_test, cnn_test_preds, pos_label=1, average='binary')
cnn_recall = recall_score(y_test, cnn_test_preds, pos_label=1, average='binary')
cnn_f1 = f1_score(y_test, cnn_test_preds, pos_label=1, average='binary')
cnn_auc = roc_auc_score(y_test, cnn_test_probs)

print("1D-CNN Evaluation Metrics:")
print(f"Accuracy: {cnn_accuracy}")
print(f"Precision: {cnn_precision}")
print(f"Recall: {cnn_recall}")
print(f"F1 Score: {cnn_f1}")
print(f"AUC: {cnn_auc}")

# Plot ROC Curve
fpr_cnn, tpr_cnn, _ = roc_curve(y_test, cnn_test_probs)
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_cnn, tpr_cnn, label=f'1D-CNN AUC = {roc_auc_cnn:.3f}', color='blue')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - 1D-CNN')
plt.legend(loc='lower right')
plt.show()

# Function to display the confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Bot', 'Bot'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.show()

# 1. Confusion Matrix for RNN
rnn_test_preds = (rnn_test_probs >= 0.5).astype(int)
plot_confusion_matrix(y_test, rnn_test_preds, model_name='RNN')

# 2. Confusion Matrix for Random Forest
stacked_test_preds = best_stacked_model.predict(X_test_rf)
plot_confusion_matrix(y_test, stacked_test_preds, model_name='Random Forest')

# 3. Confusion Matrix for Stacked Model
stacked_test_probs = (rnn_test_probs + rf_test_probs) / 2
stacked_test_preds = (stacked_test_probs >= 0.5).astype(int)
plot_confusion_matrix(y_test, stacked_test_preds, model_name='Stacked Model')

# 4. Confusion Matrix for 1D-CNN
cnn_test_preds = (cnn_test_probs >= 0.5).astype(int)
plot_confusion_matrix(y_test, cnn_test_preds, model_name='1D-CNN')

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_rf.values,           
    feature_names=X_train_rf.columns,          
    class_names=['Not Bot', 'Bot'],            
    mode='classification',                       
    discretize_continuous=True                   
)

test_instance = X_test_rf.iloc[-1]
last_row_instance = X_test_rf.iloc[-1:]

for index, test_instance in last_row_instance.iterrows():
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_rf.values,           
        feature_names=X_train_rf.columns,          
        class_names=['Not Bot', 'Bot'],            
        mode='classification',                     
        discretize_continuous=True                 
    )

    lime_exp = lime_explainer.explain_instance(
        data_row=test_instance.values,             
        predict_fn=best_stacked_model.predict_proba,          
        num_features=10                            
    )

    lime_exp.show_in_notebook(show_table=True)
    lime_exp.save_to_file('lime_explanation.html')
    lime_exp.as_pyplot_figure()
    plt.tight_layout()
    plt.show()


    print(f"Explaining Input Instance:")
    print("brianmosey")


    data_with_reverted = pd.read_csv('D:/Games/School Stuff/Github/THESIS2_ALGORHYTHM/processed_input_data.csv') 


    def explain_instance_with_reverted(instance, reverted_data):
        retweet_count = instance['Retweet Count']
        mention_count = instance['Mention Count']
        follower_count = instance['Follower Count']
        bot_label = instance['Bot Label']
        rnn_prob = instance['RNN_Prob']
        explanation = []

        if retweet_count < 1:
            explanation.append(f"**Retweet Count**: {retweet_count:.2f}\n"
                           f"A very low retweet count suggests limited influence or activity on the platform.")
        elif retweet_count < 5:
            explanation.append(f"**Retweet Count**: {retweet_count:.2f}\n"
                           f"A low retweet count may indicate the user is less active in sharing others' content.")
        else:
            explanation.append(f"**Retweet Count**: {retweet_count:.2f}\n"
                           f"This user engages more actively in retweeting, indicating some influence.")
    
        if mention_count < 1:
            explanation.append(f"**Mention Count**: {mention_count:.2f}\n"
                           f"This user has not engaged in conversations with others.")
        elif mention_count < 3:
            explanation.append(f"**Mention Count**: {mention_count:.2f}\n"
                           f"This user has mentioned others a few times, indicating some interaction.")
        else:
            explanation.append(f"**Mention Count**: {mention_count:.2f}\n"
                           f"This user is quite engaged, mentioning others frequently in their tweets.")
    
        if follower_count < 0.5:
            explanation.append(f"**Follower Count**: {follower_count:.4f}\n"
                           f"This indicates a relatively low follower count, suggesting limited influence.")
        elif follower_count < 1.0:
            explanation.append(f"**Follower Count**: {follower_count:.4f}\n"
                           f"This user has a moderate follower count.")
        else:
            explanation.append(f"**Follower Count**: {follower_count:.4f}\n"
                           f"This indicates a strong follower presence, often correlating with higher influence.")
    
        explanation.append(f"**Bot Label**: {bot_label:.4f}\n"
                       f"This indicates the model classifies this instance as not a bot.")
    
        explanation.append(f"**RNN Probability**: {rnn_prob:.6f}\n"
                       f"A low probability suggests that the likelihood of this user being a bot is low.")
    
        explanation.append(f"\n**Tweet Tokens Explanation**:")
        for column in instance.index:
            if column.startswith('Tweet_token_'):
                token_value = instance[column]
                token_number = column.split('_')[-1]
                reverted_word = reverted_data[f'Tweet_reverted_{token_number}'].iloc[instance.name]
                if token_value == 0:
                    explanation.append(f"  - **{column}**: {token_value} (low usage of specific terms) "
                                   f"Word: {reverted_word}")
                else:
                    explanation.append(f"  - **{column}**: {token_value} (engages with significant or trending terms) "
                                   f"Word: {reverted_word}")

        explanation.append(f"\n**Location Tokens Explanation**:")
        for column in instance.index:
            if column.startswith('Location_token_'):
                token_value = instance[column]
                token_number = column.split('_')[-1]
                reverted_word = reverted_data[f'Location_reverted_{token_number}'].iloc[instance.name]
                if token_value == 0:
                    explanation.append(f"  - **{column}**: {token_value} (low association with geographic terms) "
                                   f"Location: {reverted_word}")
                else:
                    explanation.append(f"  - **{column}**: {token_value} (associated with notable geographic locations) "
                                   f"Location: {reverted_word}")

        explanation.append(f"\n**Hashtags Tokens Explanation**:")
        for column in instance.index:
            if column.startswith('Hashtags_token_'):
                token_value = instance[column]
                token_number = column.split('_')[-1]
                reverted_word = reverted_data[f'Hashtags_reverted_{token_number}'].iloc[instance.name]
                if token_value == 0:
                    explanation.append(f"  - **{column}**: {token_value} (low association with hashtagged topics) "
                                   f"Hashtag: {reverted_word}")
                else:
                    explanation.append(f"  - **{column}**: {token_value} (engages with popular or trending hashtags) "
                                   f"Hashtag: {reverted_word}")

        conclusion = "### Conclusion:\n"
    
        if bot_label == 0 and rnn_prob < 0.1:
            conclusion += (
            "The analysis indicates that this user is classified as a non-bot, supported by low engagement metrics "
            "and a very low probability of being a bot according to the RNN model. This aligns with typical behavior "
            "of ordinary users on the platform."
        )
        elif bot_label == 1:
            conclusion += (
            "Despite being classified as a non-bot, the engagement metrics and feature values suggest some atypical "
            "behavior for an ordinary user. This may warrant further investigation to confirm the user's authenticity."
        )
        elif follower_count < 0.5:
            conclusion += (
            "This user has a low follower count, indicating limited influence. The classification as a non-bot, "
            "combined with low engagement metrics, suggests a typical ordinary user, albeit with potential for growth."
        )
        else:
            conclusion += (
            "Overall, while this user has some engagement with trending topics as indicated by the high token values, "
            "the low retweet and mention counts suggest that they are not a highly active participant. "
            "The classification as a non-bot aligns with these findings."
        )
    
        explanation.append(conclusion)
        return "\n".join(explanation)

    instance_explanation = explain_instance_with_reverted(test_instance, data_with_reverted)
    print("\nInstance Explanation:\n")
    print(instance_explanation)

    plt.figure(figsize=(12, 6))
    plt.title(f"Feature Values for Test Instance: brianmosey", fontsize=16)
    plt.bar(test_instance.index, test_instance.values, color='skyblue')
    plt.xticks(rotation=90)
    plt.ylabel("Feature Values", fontsize=14)
    plt.tight_layout()
    plt.show()

    explanation_text = explain_instance_with_reverted(test_instance, data_with_reverted)

    plt.figure(figsize=(10, 12))
    plt.title(f"Explanation for Test Instance: brianmosey", fontsize=16)
    plt.text(0.01, 0.99, explanation_text, va='top', fontsize=12, wrap=True, fontfamily='monospace')
    plt.axis('off')
    plt.tight_layout()      
    plt.show()