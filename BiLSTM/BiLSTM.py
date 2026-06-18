import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ks_2samp, wasserstein_distance, linregress, t, gaussian_kde
from scipy.integrate import simpson

# ==================================================
# 1. Load data
# ==================================================
df = pd.read_csv("data.csv")

SEQ_LEN = 25

mapping = {
    'A':[1,0,0,0],
    'T':[0,1,0,0],
    'C':[0,0,1,0],
    'G':[0,0,0,1],
    'N':[0,0,0,0]
}

def pad(seq):
    seq = str(seq).upper()
    if len(seq) > SEQ_LEN:
        seq = seq[-SEQ_LEN:]
    if len(seq) < SEQ_LEN:
        seq = "N"*(SEQ_LEN-len(seq)) + seq
    return seq

def encode(seq):
    seq = pad(seq)
    return np.array([mapping.get(b,[0,0,0,0]) for b in seq], dtype=np.float32)

X = np.stack(df["sequence"].apply(encode).values)
y = df["expression_level"].values.astype(np.float32)

non_n_ratio = np.mean(np.any(X != 0, axis=2))
print(f"Average non-N base ratio per position: {non_n_ratio:.2f}")


scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()


X_train, X_test, y_train, y_test = train_test_split(
    X, y_scaled, test_size=0.30, random_state=42
)


X_all = np.concatenate([X_train, X_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0)

print(f"All data for training: {X_all.shape[0]} samples")

# ==================================================
# BiLSTM
# ==================================================
inputs = layers.Input(shape=(SEQ_LEN, 4))
x = layers.Masking(mask_value=0.0)(inputs)


x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Dropout(0.6)(x)
x = layers.BatchNormalization()(x)

x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Dropout(0.6)(x)

# Attention
attention = layers.Dense(1, activation=None)(x)
attention = layers.Lambda(lambda z: tf.nn.softmax(z, axis=1))(attention)
x = layers.Multiply()([x, attention])
x = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(x)

# Regression head
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.7)(x)  # 提高
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(1)(x)

model = Model(inputs, outputs)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4,
    decay_steps=1000,
    decay_rate=0.9
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=["mae"]
)
model.summary()


history = model.fit(
    X_all, y_all,
    epochs=300,
    batch_size=32,
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)]
)


y_pred_scaled = model.predict(X_test).flatten()
y_test_raw = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_raw = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()


r2 = r2_score(y_test_raw, y_pred_raw)
pearson = pearsonr(y_test_raw, y_pred_raw)[0]
ks = ks_2samp(y_test_raw, y_pred_raw)
emd = wasserstein_distance(y_test_raw, y_pred_raw)


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12


plt.figure(figsize=(4, 4))


sns.scatterplot(x=y_test_raw, y=y_pred_raw, color='#75c074', alpha=0.4)



slope, intercept, r_value, p_value, std_err = linregress(y_test_raw, y_pred_raw)
x_sorted = np.sort(y_test_raw)
y_fit = slope * x_sorted + intercept

n = len(y_test_raw)
x_mean = np.mean(y_test_raw)
residuals = y_pred_raw - (slope * y_test_raw + intercept)
se_y = np.sqrt(np.sum(residuals**2) / (n - 2))
se_fit = se_y * np.sqrt(1/n + (x_sorted - x_mean)**2 / np.sum((y_test_raw - x_mean)**2))
t_crit = t.ppf(0.975, df=n-2)
ci_lower = y_fit - t_crit * se_fit
ci_upper = y_fit + t_crit * se_fit

plt.plot(x_sorted, y_fit, color='#75c074', linewidth=1)
plt.fill_between(x_sorted, ci_lower, ci_upper, color='#75c074', alpha=0.3)

plt.legend(title=f'R² = {r2:.2f}', title_fontsize=12, fontsize=12)
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.tight_layout()
plt.savefig("scatter.pdf")
plt.show()


def smooth_cdf(data, n_points=1000):
    kde = gaussian_kde(data)
    x_grid = np.linspace(data.min(), data.max(), n_points)
    density = kde.evaluate(x_grid)
    dx = x_grid[1] - x_grid[0]
    cdf = np.cumsum(density) * dx
    cdf = cdf / cdf[-1]
    return x_grid, cdf

plt.figure(figsize=(6, 3))
x_true_cdf, y_true_cdf = smooth_cdf(y_test_raw)
x_pred_cdf, y_pred_cdf = smooth_cdf(y_pred_raw)

plt.plot(x_true_cdf, y_true_cdf, linewidth=1, label="True value")
plt.plot(x_pred_cdf, y_pred_cdf, linewidth=1, label="Predicted value")
plt.xlabel("Value")
plt.ylabel("CDF")
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("cdf.pdf")
plt.show()


bins = np.linspace(
    min(y_test_raw.min(), y_pred_raw.min()),
    max(y_test_raw.max(), y_pred_raw.max()),
    100
)
ht, _ = np.histogram(y_test_raw, bins=bins, density=True)
hp, _ = np.histogram(y_pred_raw, bins=bins, density=True)
ct = np.cumsum(ht) / np.sum(ht)
cp = np.cumsum(hp) / np.sum(hp)
cdf_area = simpson(np.abs(ct - cp), dx=bins[1] - bins[0])   # 注意加上 dx
print("CDF area diff =", cdf_area)