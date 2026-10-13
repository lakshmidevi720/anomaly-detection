from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, Model

# Reproducibility and CPU-friendly settings
tf.keras.utils.set_random_seed(42)

ROOT = Path.home() / "Downloads" / "mri_brain_data"
OUT = Path.home() / "Downloads" / "brain_mri_models"
OUT.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 128
BATCH_SIZE = 16
LATENT_DIM = 64
EPOCHS = 15

train_dir = ROOT / "Training" / "notumor"
test_normal_dir = ROOT / "Testing" / "notumor"

def load_images(folder):
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(folder.glob(ext))
    paths = sorted(paths)
    if not paths:
        raise RuntimeError(f"No images found in {folder}")

    images = []
    for path in paths:
        image = tf.keras.utils.load_img(
            path, target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale"
        )
        images.append(tf.keras.utils.img_to_array(image) / 255.0)

    return np.asarray(images, dtype=np.float32)

print("Loading normal training images...")
x_train = load_images(train_dir)
print("Training images:", len(x_train))

print("Loading normal test images...")
x_normal_test = load_images(test_normal_dir)
print("Normal test images:", len(x_normal_test))

# Encoder
inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))
x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Conv2D(128, 3, strides=2, padding="same", activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)

z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)

def sample_latent(args):
    mean, log_var = args
    epsilon = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * epsilon

z = layers.Lambda(sample_latent, name="sampling")([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = layers.Input((LATENT_DIM,))
x = layers.Dense(16 * 16 * 128, activation="relu")(latent_inputs)
x = layers.Reshape((16, 16, 128))(x)
x = layers.Conv2DTranspose(128, 3, strides=2, padding="same",
                           activation="relu")(x)
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same",
                           activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same",
                           activation="relu")(x)
outputs = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
decoder = Model(latent_inputs, outputs, name="decoder")

class VAE(Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=False):
        mean, log_var, sample = self.encoder(inputs, training=training)
        return self.decoder(sample, training=training)

    @property
    def metrics(self):
        return [self.loss_tracker, self.recon_tracker, self.kl_tracker]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            mean, log_var, sample = self.encoder(data, training=True)
            reconstruction = self.decoder(sample, training=True)
            recon_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=(1, 2, 3))
            )
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1
                )
            )
            loss = recon_loss + kl_loss
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.loss_tracker.update_state(loss)
        self.recon_tracker.update_state(recon_loss)
        self.kl_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

print("\nStarting VAE training...")
vae.fit(
    x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1
)

# Compute normal-image reconstruction errors and select a 95th-percentile threshold.
means, log_vars, samples = encoder.predict(x_normal_test, batch_size=BATCH_SIZE)
normal_recon = decoder.predict(means, batch_size=BATCH_SIZE)
normal_errors = np.mean(
    np.square(x_normal_test - normal_recon), axis=(1, 2, 3)
)
threshold = float(np.percentile(normal_errors, 95))

encoder.save(OUT / "mri_encoder.keras")
decoder.save(OUT / "mri_decoder.keras")

settings = {
    "image_size": IMG_SIZE,
    "latent_dim": LATENT_DIM,
    "threshold": threshold,
    "threshold_method": "95th percentile of normal test reconstruction errors",
    "training_data": "Training/notumor",
    "threshold_calibration_data": "Testing/notumor",
    "note": "Research prototype; threshold requires independent validation."
}
with open(OUT / "model_settings.json", "w") as f:
    json.dump(settings, f, indent=2)

print("\nTraining finished.")
print("Models and settings saved in:", OUT)
print("Threshold:", threshold)
print("Mean normal reconstruction error:", float(normal_errors.mean()))
print("This model is for research, not clinical diagnosis.")
