# Özellikler
Veri Seti: Monkeypox ve diğer hastalıkları içeren geniş bir görüntü veri seti kullanılmıştır.
Derin Öğrenme Modeli: VGG16 modeli, özellik çıkarma için kullanılmış ve eğitim veri setindeki görüntülerden özellik vektorleri elde edilmiştir.
SVM (Destek Vektör Makineleri): Elde edilen özellik vektorleri, SVM kullanılarak sınıflandırılmıştır.
Özellik Seçimi: SelectKBest yöntemiyle en iyi özellikler seçilmiş ve model performansı artırılmıştır.
 
 # SVM ile Görüntü Sınıflandırma

Bu proje, Support Vector Machines (SVM) kullanarak monkeypox hastalığı görüntülerini benzer hastalıklardan ayırmak için sınıflandırma modeli oluşturmayı gösterir. Proje, özellikle VGG16 modeli kullanılarak elde edilen özellik vektörleri üzerinde çalışır ve SVM ile bu özellik vektörlerini kullanarak sınıflandırma yapar.

## Veri Seti

Bu projede kullanılan veri seti "Fold1" adlı bir klasör içindedir. Bu klasörün altında "Train" ve "Test" klasörleri bulunmaktadır.

## Accuracy

Modelin eğitim ve test doğruluk değeri 0.9111111111111111"dur. Hiç de fena değil :D
