import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# -----------------------------
# Configuration Parameters
# -----------------------------
NUM_CYCLES = 64
INPUT_DIM = 64
HIDDEN_DIM = 64
OUTPUT_DIM = 2
MAX_EPOCHS_PER_IMAGE = 50  # maximum repeated passes on a single image if wrong

# Weight update factors (for low-magnitude features)
WEIGHT_HEBBIAN = 0.3
WEIGHT_DECAY = 0.2
WEIGHT_REINFORCEMENT = 0.5
WEIGHT_PUNISHMENT = 0.2

# Threshold adjustments (kept small so accumulation can reach the threshold)
THRESHOLD_HEBBIAN = 0.2
THRESHOLD_REINFORCEMENT = 0.4
THRESHOLD_PUNISHMENT = 0.3

DECAY_RATE = 0.02
INITIAL_BUCKET_THRESHOLD = 1.5  # bucket fires when accumulated value reaches 1.5
SLEEP_THRESHOLD = 0.01
SLEEP_FACTOR = 0.1

# -----------------------------
# CNN for Feature Extraction
# -----------------------------
cnn_input_shape = (64, 64, 1)
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=cnn_input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(INPUT_DIM, activation='relu')
])
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy')
cnn_model.summary()

# -----------------------------
# Data Generators and Preloading
# -----------------------------
data_dir = 'dataset_directory'
train_data_dir = os.path.join(data_dir, 'training_set')
test_data_dir = os.path.join(data_dir, 'test_set')

datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=1,
    class_mode='categorical',
    shuffle=False  # disable shuffle to later alternate manually
)
test_generator = datagen.flow_from_directory(
    test_data_dir,
    target_size=(64, 64),
    color_mode='grayscale',
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Preload all training images and labels
train_images = []
train_labels = []
num_train = train_generator.n
for i in range(num_train):
    img, label = train_generator.next()
    train_images.append(img)
    train_labels.append(label)

# Partition into two classes (assuming two classes: 0 and 1)
class0 = []
class1 = []
for img, label in zip(train_images, train_labels):
    if np.argmax(label[0]) == 0:
        class0.append((img, label))
    else:
        class1.append((img, label))

# Create an alternating training list to help prevent overfitting
alternating_train = []
min_len = min(len(class0), len(class1))
for i in range(min_len):
    alternating_train.append(class0[i])
    alternating_train.append(class1[i])
# Append leftover images if classes are imbalanced
if len(class0) > min_len:
    alternating_train.extend(class0[min_len:])
elif len(class1) > min_len:
    alternating_train.extend(class1[min_len:])

# -----------------------------
# Initialize ANN Parameters
# -----------------------------
W_in_hidden = np.random.randn(NUM_CYCLES, HIDDEN_DIM) * 0.1
bucket_thresholds = np.full((HIDDEN_DIM,), INITIAL_BUCKET_THRESHOLD)
bucket_fill = np.zeros((HIDDEN_DIM,))
node_toll = np.ones((HIDDEN_DIM,))
W_hidden_out = np.random.randn(HIDDEN_DIM, OUTPUT_DIM) * 0.1

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict_output(hidden_activation):
    logits = np.dot(hidden_activation, W_hidden_out)
    probs = softmax(logits)
    return probs

def custom_ann_update(feature_vector, true_label):
    """
    Processes one image's 64-dimensional feature vector over 64 cycles.
    Immediate Hebbian learning and decay occur in each cycle.
    After one complete 64-cycle pass, reward or punishment is applied.
    A sleep (residual removal) mechanism is applied after the full cycle.
    
    Returns:
        predicted_class (int) and a temporary log of per-cycle updates.
    """
    global W_in_hidden, bucket_thresholds, bucket_fill, node_toll, W_hidden_out
    temp_cycle_info = []
    firing_counts = np.zeros((HIDDEN_DIM,))
    
    true_class = np.argmax(true_label)
    desired_sign = -1 if true_class == 0 else 1

    # Process each cycle
    for cycle in range(NUM_CYCLES):
        feature = feature_vector[cycle]
        for i in range(HIDDEN_DIM):
            # Accumulate weighted feature with decay
            bucket_fill[i] += W_in_hidden[cycle, i] * feature
            bucket_fill[i] *= (1 - DECAY_RATE * WEIGHT_DECAY)
            
            # Fire if bucket value reaches its threshold
            if bucket_fill[i] >= bucket_thresholds[i]:
                signal = bucket_fill[i]
                # Immediate Hebbian update for this cycle
                W_in_hidden[cycle, i] += WEIGHT_HEBBIAN * feature
                bucket_thresholds[i] -= THRESHOLD_HEBBIAN
                
                # Apply reinforcement or punishment based on the sign after full cycle
                if (signal * desired_sign) > 0:
                    W_in_hidden[cycle, i] += WEIGHT_REINFORCEMENT * feature
                    bucket_thresholds[i] -= THRESHOLD_REINFORCEMENT
                    update_type = 'reinforcement'
                else:
                    W_in_hidden[cycle, i] -= WEIGHT_PUNISHMENT * feature
                    bucket_thresholds[i] += THRESHOLD_PUNISHMENT
                    update_type = 'punishment'
                
                firing_counts[i] += 1
                temp_cycle_info.append((cycle, i, signal, update_type))
                bucket_fill[i] = 0  # reset bucket after firing

    # Compute output based on firing counts from all cycles
    hidden_activation = firing_counts
    logits = np.dot(hidden_activation, W_hidden_out)
    probs = softmax(logits)
    predicted_class = np.argmax(probs)
    
    # After the full cycle, adjust output weights based on the cycle logs
    for (cycle, i, signal, update_type) in temp_cycle_info:
        if update_type == 'reinforcement':
            W_hidden_out[i, true_class] += WEIGHT_REINFORCEMENT * signal
        elif update_type == 'punishment':
            W_hidden_out[i, predicted_class] -= WEIGHT_PUNISHMENT * signal
    
    # Sleep (residual removal) mechanism to clear small bucket fills
    if np.all(np.abs(bucket_fill) < SLEEP_THRESHOLD):
        for i in range(HIDDEN_DIM):
            adjustment = (bucket_fill[i] / node_toll[i]) * SLEEP_FACTOR
            for cycle in range(NUM_CYCLES):
                W_in_hidden[cycle, i] += adjustment
            bucket_fill[i] = 0

    return predicted_class, temp_cycle_info

# -----------------------------
# Training Loop (Alternating Images)
# -----------------------------
print("Starting training on alternating images...")
for img_idx, (img, label) in enumerate(alternating_train):
    true_label = label[0]
    true_class = np.argmax(true_label)
    cnn_feature = cnn_model.predict(img)[0]
    
    bucket_fill = np.zeros((HIDDEN_DIM,))  # reset bucket fill for this image
    epoch = 0
    correct = False
    # Continue on the same image until the correct class is obtained
    while not correct and epoch < MAX_EPOCHS_PER_IMAGE:
        pred_class, cycle_log = custom_ann_update(cnn_feature, true_label)
        if pred_class == true_class:
            correct = True
            print(f"Image {img_idx+1}/{len(alternating_train)} classified correctly in {epoch+1} epoch(s).")
        else:
            epoch += 1
    bucket_fill = np.zeros((HIDDEN_DIM,))  # clear residual activation

print("Training complete.")

# -----------------------------
# Evaluation Loop
# -----------------------------
print("Starting evaluation on test set...")
num_test = test_generator.n
correct_count = 0
for i in range(num_test):
    img, label = test_generator.next()
    true_label = label[0]
    true_class = np.argmax(true_label)
    cnn_feature = cnn_model.predict(img)[0]
    firing_counts = np.zeros((HIDDEN_DIM,))
    fill_temp = np.zeros((HIDDEN_DIM,))
    for cycle in range(NUM_CYCLES):
        feature = cnn_feature[cycle]
        for j in range(HIDDEN_DIM):
            fill_temp[j] += W_in_hidden[cycle, j] * feature
            fill_temp[j] *= (1 - DECAY_RATE * WEIGHT_DECAY)
            if fill_temp[j] >= bucket_thresholds[j]:
                firing_counts[j] += 1
                fill_temp[j] = 0
    hidden_activation = firing_counts
    logits = np.dot(hidden_activation, W_hidden_out)
    probs = softmax(logits)
    pred_class = np.argmax(probs)
    if pred_class == true_class:
        correct_count += 1

print(f"Test Accuracy: {correct_count / num_test * 100:.2f}%")
