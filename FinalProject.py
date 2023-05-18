#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
import seaborn as sns
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, Embedding, TextVectorization, concatenate, Lambda, BatchNormalization
from keras import Input, backend
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, Callback
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam



#%% Import dataset
df = pd.read_csv('Real_Estate_Sales_2001-2020_GL.csv')
print(df.info())
df.describe().T
#%% Visualizing the correlations between numerical variables
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="RdBu")
plt.title("Correlations Between Variables", size=15)
plt.show()
#%%
print("Missing Values by Column")
print("-"*30)
print(df.isna().sum())
print("-"*30)
print("TOTAL MISSING VALUES:",df.isna().sum().sum())
#%% Pre-Processing data
df = df.iloc[:,1:10] #Drop 5 feature
df = df.iloc[:,df.columns != 'Date Recorded']
df = df.iloc[:,df.columns != 'Address']
df['Residential Type'] = df['Residential Type'].fillna('Non-Residential')

df = df.dropna() # Drop Null value
df = df[df['Sales Ratio'] > 0.2]
df = df.iloc[:,df.columns != 'Sales Ratio']
print(df.info())
# Get count duplicates single column using dataframe.pivot_table()
duplicates = df.pivot_table(index = ['Residential Type'], aggfunc ='size')
print(duplicates)
#One-hot encoding
df = pd.get_dummies(df, columns=['Property Type', 'Residential Type'])

# %% Scale data
prices_scaler = StandardScaler()
assessed_prices_scaler = StandardScaler()
years_scaler = StandardScaler()

df['Sale Amount'] = prices_scaler.fit_transform(df['Sale Amount'].values.reshape(-1,1))
df['Assessed Value'] = assessed_prices_scaler.fit_transform(df['Assessed Value'].values.reshape(-1,1))
df['List Year'] = assessed_prices_scaler.fit_transform(df['List Year'].values.reshape(-1,1))

# %% Split to test and train 
x_full = df.iloc[:,df.columns != 'Sale Amount'].values
y_full = df.iloc[:,df.columns == 'Sale Amount'].values
x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, shuffle=True)

# %% Preprocess text to be passed to an Embedding layer.
def find_max_len(vocab):
    max_len = 0
    for string in vocab:
        if max_len < len(string.split()):
            max_len = len(string.split())
    return max_len

town_vocab = np.unique(df['Town'])
town_vec_dim = int(math.sqrt(len(town_vocab)))
town_max_len = find_max_len(town_vocab)
town_text_vectorizer = TextVectorization(max_tokens=len(town_vocab), output_mode="int", output_sequence_length=town_max_len)
town_text_vectorizer.adapt(town_vocab)

# address_vocab = np.unique(df['Address'])
# address_vec_dim = int(math.sqrt(len(address_vocab)))
# address_max_len = find_max_len(address_vocab)
# address_text_vectorizer = TextVectorization(max_tokens=len(address_vocab), output_mode="int", output_sequence_length=address_max_len)
# address_text_vectorizer.adapt(address_vocab)

# %% Create NN model
num_inputs = Input(shape=(18), dtype=tf.float32)

town_inputs = Input(shape=(1), dtype=tf.string)
town_embedding_layer = Embedding(input_dim=town_text_vectorizer.vocabulary_size() + 1, input_length=town_max_len, output_dim=town_vec_dim, name='Town_Embedding_Layer')
#Preprocess the string inputs, turning them into int sequences
town_sequences = town_text_vectorizer(town_inputs)
town_embed = town_embedding_layer(town_sequences)
town_flatten = Flatten()(town_embed)

# address_inputs = Input(shape=(1), dtype=tf.string)
# address_embedding_layer = Embedding(input_dim=address_text_vectorizer.vocabulary_size() + 1, input_length=address_max_len, output_dim=address_vec_dim, name='Address_Embedding_Layer')
# #Preprocess the string inputs, turning them into int sequences
# address_sequences = Lambda(address_text_vectorizer)(address_inputs)
# address_embed = address_embedding_layer(address_sequences)
# address_flatten = Flatten()(address_embed)

concaten = concatenate([num_inputs, town_flatten])

batch_norm1 = BatchNormalization()(concaten)
hidden1 = Dense(batch_norm1.shape[1], activation='relu')(batch_norm1)
drop_out1 = Dropout(0.2)(hidden1)
hidden2 = Dense(batch_norm1.shape[1],activation='relu')(drop_out1)
drop_out2 = Dropout(0.2)(hidden2)
hidden3 = Dense(batch_norm1.shape[1],activation='relu')(drop_out2)
drop_out3 = Dropout(0.2)(hidden3)
hidden4 = Dense(batch_norm1.shape[1],activation='relu')(drop_out3)
drop_out4 = Dropout(0.2)(hidden3)
hidden5 = Dense(batch_norm1.shape[1], activation='relu')(drop_out4)
drop_out5 = Dropout(0.2)(hidden5)
output = Dense(1)(drop_out5)

model = Model(inputs=[num_inputs, town_inputs], outputs=[output])
model.save('house_sale_fresh_model')
#model.save('house_data_fresh')
model.summary()
plot_model(model, "house_data_data.png", show_shapes=True)
#%% Prepare for Model Input
data_train = (np.asarray(np.c_[x_train[:,0], x_train[:,3:]]).astype(np.float32), x_train[:,1].reshape((x_train[:,1].shape[0],1)))
data_test = (np.asarray(np.c_[x_test[:,0], x_test[:,3:]]).astype(np.float32), x_test[:,1].reshape((x_test[:,1].shape[0],1)))
#%% Find good learning rate
def qFindLearningRate(model, X_train, y_train, increase_factor = 1.0005, batch_size=128, fig_name='find_lr'):
    # Create a callback to increase the learning rate after each batch, store losses to plot later
    class IncreaseLearningRate_cb(Callback):
        def __init__(self, factor):
            self.factor = factor
            self.rates = []
            self.losses = []
        def on_batch_end(self, batch, logs):
            K = backend
            self.rates.append(K.get_value(self.model.optimizer.learning_rate))
            self.losses.append(logs["loss"])
            K.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)
    increase_lr = IncreaseLearningRate_cb(factor=increase_factor)

    # Train epoch
    history = model.fit(X_train, np.asarray(y_train).astype(np.float32), epochs=1, batch_size=batch_size, callbacks=[increase_lr])

    # Plot losses after training batches. 
    # NOTE: a batch has a different learning rate 
    from statistics import median
    plt.plot(increase_lr.rates, increase_lr.losses)
    plt.gca().set_xscale('log')
    #plt.hlines(min(increase_lr.losses), min(increase_lr.rates), max(increase_lr.rates))
    plt.axis([min(increase_lr.rates), max(increase_lr.rates), min(increase_lr.losses), median(increase_lr.losses)])
    plt.grid()
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.savefig(fig_name, dpi=300)
    plt.show()
    return increase_lr
model = load_model('house_sale_fresh_model')
model.compile(loss='mse', optimizer=Adam(learning_rate=1e-3), metrics=[RootMeanSquaredError()])
increase_lr = qFindLearningRate(model, data_train, y_train, fig_name='find_lr', batch_size=512)
# %% Compile and Train model
model = load_model('house_sale_fresh_model')
model.compile(loss='mse', optimizer=Adam(learning_rate=1e-3), metrics=[RootMeanSquaredError()])
model.run_eagerly = True

early_stop = EarlyStopping(monitor='val_loss', patience=2)
# model_cp = tf.keras.callbacks.ModelCheckpoint(
#     filepath='best_model',
#     monitor='val_loss',
#     verbose=1,
#     save_best_only=True,
#     save_format='tf'
# )

history = model.fit(data_train, np.asarray(y_train).astype(np.float32), epochs=12, 
                    validation_split=0.2,
                    batch_size= 1024,
                    callbacks=[early_stop])

model.save('house_sale_model')
#%% Plot losses
model = load_model('house_sale_model')
val_loss = history.history['val_loss']
loss = history.history['loss']
plt.plot(range(len(val_loss)),val_loss,'c',label='Validation loss')
plt.plot(range(len(loss)),loss,'m',label='Train loss')

plt.title('Training and validation losses')
plt.legend()
plt.xlabel('epochs')
plt.show()
# %%
model = load_model('house_sale_model')
predicted_price = model.predict(data_test)

print(prices_scaler.inverse_transform(y_test)[:10])
print(prices_scaler.inverse_transform(predicted_price)[:10])

plt.plot(prices_scaler.inverse_transform(predicted_price[:100]),color='r')
plt.plot(prices_scaler.inverse_transform(y_test[:100]))
plt.legend()
plt.show()
# %%
def plot_predictions(y_pred, y_true, title):
    plt.style.use('ggplot')  # optional, that's only to define a visual style
    plt.scatter(y_pred, y_true, s=10, alpha=0.5)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(title)

plot_predictions(prices_scaler.inverse_transform(predicted_price)[:500], prices_scaler.inverse_transform(y_test)[:500], title='Predictions on the training set')
# %%
def plot_history(metrics):
    """
    Plot the training history

    Args:
        metrics(str, list): Metric or a list of metrics to plot
    """
    history_df = pd.DataFrame.from_dict(history.history)
    sns.lineplot(data=history_df[metrics])
    plt.xlabel("epochs")
    plt.ylabel("RMSE")

plot_history('root_mean_squared_error')