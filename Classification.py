# Gerekli kütüphaneleri yükleyin
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Veri setinin bulunduğu ana klasör
data_folder = 'C:/Users/lenovo/Downloads/Augmented Images/Augmented Images'


# Sınıflar
classes = ['Monkeypox_augmented', 'Others_augmented']  # 'Monkey Pox' için '0', 'Others' için '1'

# Özellik vektörleri ve etiketleri depolamak için boş listeler
X_data, y_data = [], []

# Her sınıftaki verileri yükle
for class_name in classes:
    class_folder = os.path.join(data_folder, class_name)

    # Her bir veri dosyasını yükle
    for file_name in os.listdir(class_folder):
        if file_name.endswith('.npy'):
            file_path = os.path.join(class_folder, file_name)

            # Özellik vektörleri ve etiketleri yükle
            features = np.load(file_path)
            labels = np.full(len(features), int(class_name))

            # Listelere ekle
            X_data.extend(features)
            y_data.extend(labels)

# Veriyi karıştırın
combined = list(zip(X_data, y_data))

# Kontrol et: Eğer veri seti boşsa, hatayı önleyin
if combined:
    np.random.shuffle(combined)
    X_data[:], y_data[:] = zip(*combined)
else:
    print("Veri seti boş, karıştırma işlemi yapılamıyor.")

# Veriyi eğitim ve test kümelerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)



# Sınıf ağırlıklarını belirleme
class_weights = {0: 1.0, 1: 2.0}  # Bu örnekte, 'Monkey Pox' sınıfı için 1.0, 'Others' sınıfı için 2.0

# Destek Vektör Makineleri (SVM) sınıflandırma modelini oluşturun
model = SVC(kernel='linear', class_weight=class_weights, random_state=42)

# Modeli eğitin
model.fit(X_train, y_train)

# Test kümesi üzerinde modelin performansını değerlendirin
y_pred = model.predict(X_test)

# Sınıflandırma raporu ve karışıklık matrisi
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
