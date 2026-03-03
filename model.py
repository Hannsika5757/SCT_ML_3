import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# =========================
# SETTINGS
# =========================
train_dir = "train/train"
test_dir = "test/test"
IMAGE_SIZE = 64
TOTAL_IMAGES = 10000   # Use 10,000 images

# =========================
# LOAD TRAIN DATA
# =========================
images = []
labels = []

cat_count = 0
dog_count = 0
limit_per_class = TOTAL_IMAGES // 2

print("Loading training images...")

for filename in os.listdir(train_dir):

    if not filename.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(train_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    if "cat" in filename.lower() and cat_count < limit_per_class:
        label = 0
        cat_count += 1
    elif "dog" in filename.lower() and dog_count < limit_per_class:
        label = 1
        dog_count += 1
    else:
        continue

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # =========================
    # HOG FEATURE EXTRACTION
    # =========================
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    images.append(features)
    labels.append(label)

    if cat_count >= limit_per_class and dog_count >= limit_per_class:
        break

X = np.array(images)
y = np.array(labels)

print("Total Images Used:", X.shape)

# =========================
# SCALE FEATURES
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TUNED SVM (Optimized)
# =========================
print("Training SVM...")

model = SVC(
    kernel='rbf',
    C=5,
    gamma='scale'
)

model.fit(X_train, y_train)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)

print("Validation Accuracy:", accuracy)

# =========================
# LOAD TEST DATA
# =========================
print("Loading test images...")

test_features = []
test_ids = []

for filename in os.listdir(test_dir):

    if not filename.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(test_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    test_features.append(features)
    test_ids.append(int(filename.split('.')[0]))

X_test = np.array(test_features)
X_test = scaler.transform(X_test)

print("Predicting...")

predictions = model.predict(X_test)

submission = pd.DataFrame({
    "id": test_ids,
    "label": predictions
})

submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully!")