import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, RMSprop, AdamW
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE


# Load data from Excel
df = pd.read_excel("consolidated_all_india.xlsx")

# Exploratery data analysis
print('Data type', df.dtypes)

#data = data.fillna(0)

nan_columns = df.isna().sum()
print(nan_columns)

null_columns = df.isnull().sum()
print(null_columns)

# Check for inf, -inf, and extremely large values
inf_values = df.isin([np.inf, -np.inf])
large_values = df > np.finfo(np.float64).max

# Print the summary of where inf or extremely large values are found
print("Columns with inf or -inf values:", inf_values.any())
print("Columns with extremely large values:", large_values.any())

# Replace inf and -inf with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Optionally, replace extremely large values with NaN
df[large_values] = np.nan

# Replace all the NaN values using forword-fill method
df.fillna(0, inplace=True)

# Verify that there are no more inf, -inf, or NaN values
print(df.isin([np.inf, -np.inf]).sum().sum())  # Should be 0 if all inf/-inf are replaced
print(df.isnull().sum().sum())  # Should be 0 if all NaN values are replaced


# Assuming the target column is 'under_5_mortality' and the rest are features
X = df.drop(columns=['0-4_infant_mortality_rate'])
y = df['0-4_infant_mortality_rate']

target_binary = np.where(y != 0, 1, 0)



y_binary = (target_binary > 0.5).astype(int)

# Feature scaling
def apply_feature_scaling(df, scaling_method='standard'):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaling method. Choose 'standard', 'minmax', or 'robust'.")

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

x_scaled = apply_feature_scaling(X, scaling_method='minmax')
x_scaled

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_binary, test_size=0.3, random_state=42)

# # Oversampling
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# # feature selection
# selector = SelectKBest(score_func=f_classif, k=30)
# selector.fit(X_train_resampled, y_train_resampled)

# selected_columns = X_train.columns[selector.get_support()]

# X_train_selected = selector.transform(X_train_resampled)
# X_test_selected = selector.transform(X_test)

# # Feature Scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Convert back to DataFrame for feature selection with column names
# X_train_df = pd.DataFrame(X_train, columns=X.columns)
# X_test_df = pd.DataFrame(X_test, columns=X.columns)

# # Feature Selection
# selector = SelectKBest(score_func=f_classif, k=30)
# selector.fit(X_train, y_train)
# selected_columns = X_train.columns[selector.get_support()]

# X_train_selected = selector.transform(X_train)
# X_test_selected = selector.transform(X_test)

# # # Random Forest
# model = RandomForestRegressor(random_state = 42)

# # Fit the model to the training data
# model.fit(X_train_df, y_train)

# # Calculate permutation importance
# feature_importance = permutation_importance(model, X_test_df, y_test, n_repeats=10, random_state=42)

# # Create a DataFrame for visualization
# importance_df = pd.DataFrame({
#     'Feature': X_train_df.columns,
#     'Importance': feature_importance.importances_mean
# })

# # Sort features by importance
# importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# # Drop features with importance below a certain threshold (e.g., importance < 0.01)
# threshold = 0.01
# important_features = importance_df[importance_df['Importance'] >= threshold]['Feature'].tolist()

# # Select only important features
# X_train_selected = X_train_df[important_features]
# X_test_selected = X_test_df[important_features]

# # Convert the selected features to NumPy arrays before reshaping
# X_train_selected = X_train_selected.values
# X_test_selected = X_test_selected.values

# # Reshape the input for the CNN
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Reshape the input for the CNN
# X_train_selected = X_train_resampled.values.reshape(X_train_resampled.shape[0], X_train_resampled.shape[1], 1)
# X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

X_train_selected = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)

# Model Architecture
model = models.Sequential()

# 1D Convolutional Layers
model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_selected.shape[1], 1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dropout(rate=0.2))

model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dropout(rate=0.3))

# Adding one more Conv1D Layer
model.add(layers.Conv1D(filters=512, kernel_size=3, activation='relu'))  # New Layer
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Dropout(rate=0.4))

# Flatten the output
model.add(layers.Flatten())

# Fully Connected Layer
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(rate=0.3))  # Dropout layer
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(rate=0.3))

# Output Layer
model.add(layers.Dense(1, activation='sigmoid'))

# # **Adding class weights** to handle class imbalance
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
# class_weights = dict(enumerate(class_weights))

# Experiment with a different optimizer
#optimizer = Adam(learning_rate=0.1)  # Can try RMSprop or Adam

# Compile the model
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

# Train the model
history = model.fit(X_train_selected, y_train, epochs=30,
                     batch_size=32, validation_split=0.3 )#, class_weight=class_weights)

# Evaluate the model
results = model.evaluate(X_test, y_test)

# Predictions
#predictions = model.predict(X_test)

# Print model summary
model.summary()

y_pred_proba = model.predict(X_test)

predictions_binary = (y_pred_proba >= 0.5).astype(int) 

#precision = precision_score(y_test, y_pred_adjusted)
#predictions_binary = (predictions >= 0.5).astype(int)  # Convert probabilities to binary outcomes

# Evaluation Metrics
accuracy = accuracy_score(y_test, predictions_binary)
precision = precision_score(y_test, predictions_binary)
recall = recall_score(y_test, predictions_binary)
f1 = f1_score(y_test, predictions_binary)
roc_auc = roc_auc_score(y_test, predictions_binary)

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, predictions_binary)
class_report = classification_report(y_test, predictions_binary)

# Print evaluation results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
