import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, ks_2samp, wasserstein_distance, gaussian_kde, linregress, t
from scipy.integrate import simps
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# =========================
# 1. data
# =========================
df = pd.read_csv("data.csv")

codons = df["sequence"].str[-3:]

print(Counter(codons))
print(df.columns.tolist())
print(df.shape)

print(df["expression_level"].describe())
lengths = df["sequence"].str.len()

print(lengths.describe())

print(lengths.value_counts().sort_index())
plt.hist(df["expression_level"], bins=50)
plt.show()
dup = df.groupby("sequence")["expression_level"].agg(
    ["count","mean","std"]
)

dup = dup[dup["count"]>1]

print(len(dup))
print(dup.head())

SEQ_LEN = 25

# =========================
# SAFE mapping
# =========================
mapping = {
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'C': [0, 0, 1, 0],
    'G': [0, 0, 0, 1],
    '0': [0, 0, 0, 0]
}

# =========================
# SAFE padding
# =========================
def pad(seq):
    seq = str(seq).upper()
    seq = seq[:SEQ_LEN]
    if len(seq) < SEQ_LEN:
        seq = seq + "N" * (SEQ_LEN - len(seq))
    return seq


def encode(seq):
    seq = pad(seq)
    x = np.zeros((SEQ_LEN, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in mapping:
            x[i] = mapping[b]
        else:
            x[i] = [0, 0, 0, 0]
    return x

# =========================
# build dataset
# =========================
X = np.stack(df["sequence"].apply(encode).values)
y = df["expression_level"].values.astype(np.float32)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


X_all = np.concatenate([X_train, X_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0)

# =========================
# 2. model (stable RBS CNN)
# =========================
inputs = layers.Input(shape=(25, 4))

# CNN backbone
x = layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
x = layers.Conv1D(128, 5, padding="same", activation="relu")(x)
x = layers.Conv1D(64, 7, padding="same", activation="relu")(x)

# attention pooling
att = layers.Dense(1, activation="tanh")(x)
att = tf.nn.softmax(att, axis=1)
x = tf.reduce_sum(x * att, axis=1)

# regression head
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(1)(x)

model = Model(inputs, outputs)


model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0035, momentum=0.9),
    loss=tf.keras.losses.Huber(),
    metrics=["mae"]
)

# =========================
# 3. training
# =========================
history = model.fit(
    X_all, y_all,
    epochs=250,
    batch_size=16,
    verbose=1
)

# =========================
# 4. prediction
# =========================
y_pred = model.predict(X_test).flatten()

y_test_raw = y_test
y_pred_raw = y_pred

# =========================
# 5. metrics
# =========================
from scipy.stats import wasserstein_distance
r2 = r2_score(y_test_raw, y_pred_raw)
pearson = pearsonr(y_test_raw, y_pred_raw)[0]
ks = ks_2samp(y_test_raw, y_pred_raw)
emd = wasserstein_distance(y_test_raw, y_pred_raw)



plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

# =========================
# 6. scatter plot (with regression line + confidence band)
# =========================
plt.figure(figsize=(4, 4))
sns.scatterplot(x=y_test_raw, y=y_pred_raw, color='#db5f52',alpha=0.4)
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
plt.plot(x_sorted, y_fit, color='#db5f52', linewidth=1)
plt.fill_between(x_sorted, ci_lower, ci_upper, color='#db5f52', alpha=0.3)
plt.legend(title=f'R² = {r2:.2f}')
plt.xlabel("True value")
plt.ylabel("Predicted value")
plt.tight_layout()
plt.savefig("scatter.pdf")
plt.show()


# =========================
# 7. CDF plot (manual, compatible)
# =========================
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
plt.legend()
plt.tight_layout()
plt.savefig("cdf.pdf")
plt.show()
# =========================
# 8. CDF distance
# =========================
bins = np.linspace(
    min(y_test_raw.min(), y_pred_raw.min()),
    max(y_test_raw.max(), y_pred_raw.max()),
    100
)

ht, _ = np.histogram(y_test_raw, bins=bins, density=True)
hp, _ = np.histogram(y_pred_raw, bins=bins, density=True)

ct = np.cumsum(ht) / np.sum(ht)
cp = np.cumsum(hp) / np.sum(hp)

cdf_area = simps(np.abs(ct - cp))
print("CDF area diff =", cdf_area)