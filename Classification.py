import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder

# Veri setinin yüklenmesi ve özellik vektörlerinin oluşturulması
def load_and_preprocess_images(folder, model):
    images = []
    labels = []
    for class_folder in os.listdir(folder):
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Özellik vektörünü çıkarma
                features = model.predict(img_array)
                images.append(features.flatten())  # Özellik vektörünü düzleştirme
                labels.append(class_folder)
    return np.array(images), np.array(labels)

# Veri seti dizini
data_folder = 'C:/Users/lenovo/Downloads/Fold1/Fold1/Fold1'

# VGG16 modelini yükleyip özellik vektörlerini çıkartma
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

# Eğitim veri setini yükleme ve özellik vektörlerini çıkartma
train_data_folder = os.path.join(data_folder, 'Train')
images_train, labels_train = load_and_preprocess_images(train_data_folder, model)

# Test veri setini yükleme ve özellik vektörlerini çıkartma
test_data_folder = os.path.join(data_folder, 'Test')
images_test, labels_test = load_and_preprocess_images(test_data_folder, model)

# Etiketleri sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
encoded_labels_train = label_encoder.fit_transform(labels_train)
encoded_labels_test = label_encoder.transform(labels_test)

# Eğitim ve test setlerini oluşturma
X_train, X_test, y_train, y_test = images_train, images_test, encoded_labels_train, encoded_labels_test

# Özellik seçimi
k_best = SelectKBest(f_classif, k=150)
X_train_selected = k_best.fit_transform(X_train, y_train)
X_test_selected = k_best.transform(X_test)

# Veri standardizasyonu
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_selected)
X_test_std = scaler.transform(X_test_selected)

# Destek Vektör Makineleri (SVM) modelini oluşturma ve eğitme
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train_std, y_train)

# Test seti üzerinde modeli değerlendirme
y_pred = svm_classifier.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')

# Sınıflandırma raporu
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()