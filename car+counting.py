#OpenCV ve NumPy kütüphanelerini içe aktarıyoruz. 
import cv2
import numpy as np

# Video dosyasının yolu
video_path = "C:\\Users\\pc\\Desktop\\traffiCar\\traffic.avi"

#VideoCapture sınıfını kullanarak, belirtilen video dosyasını okumak için bir video yakalayıcısı oluşturulu
vid = cv2.VideoCapture(video_path)


# MOG2 algoritmasını kullanarak bir arka plan çıkarıcı  oluşturulur. Bu, görüntüdeki arka planı algılamak ve kaldırmak için kullanılacak.
backsub = cv2.createBackgroundSubtractorMOG2()

# Araç sayısını tutmak için bir sayaç oluşturulur.
car_count = 0

#Sonsuz bir döngü başlatılır. Bu döngü video dosyasını okur ve her kare üzerinde işlem yapar.
while True:
    # vid.read() ile bir video karesi okunur ve ret değişkeni ile başarılı bir şekilde okunup okunmadığı kontrol edilir.
    ret, frame = vid.read()
    if ret:  #Kare başarılı bir şekilde okunduysa işlemlere devam edilir.
        #Arka plan çıkarıcıyı  kullanarak, arka planı çıkarılmış bir görüntü elde edilir.
        fgmask = backsub.apply(frame)
        #Kare üzerine, araçların geçişini kontrol etmek için iki çizgi çizilir.
        cv2.line(frame, (50, 0), (50, 300), (0, 255, 0), 2)
        cv2.line(frame, (70, 0), (70, 300), (0, 255, 0), 2)
        #Arka plan çıkarılmış görüntü üzerinde konturlar (sınırlayıcı kutular) bulunur.
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try: hierarchy = hierarchy[0]
        except: hierarchy = []
     #Konturun boyutu kontrol edilir. Eğer belirli bir boyutun üzerindeyse (w > 40 and h > 40),
     #dikdörtgen çizilir ve eğer belirli bir alanda ise (if x > 50 and x < 70), araç sayacı artırılır.
        for contour, hier in zip(contours, hierarchy):
            x, y, w, h = cv2.boundingRect(contour)  #Konturun sınırlayıcı dikdörtgeni (bounding rectangle) hesaplanır.
            if w > 40 and h > 40:   #Sınırlayıcı dikdörtgenin boyutu belirli bir eşiği aşıyorsa işlemlere devam edilir.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)  #Kare üzerine sınırlayıcı dikdörtgen çizilir.
                if x > 50 and x < 70:   #Belirli bir bölgede (iki çizgi arasında) bir araç algılanırsa, araç sayacı artırılır.
                    car_count += 1
                    
    
        #  Kareye araç sayısını yazdırmak için bir metin eklenir.
        cv2.putText(frame, "car: " + str(car_count), (90, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Araç sayma işlemi gerçekleştirilen kare gösterilir.
        cv2.imshow("Car Counting", frame)
        #Arka plan çıkarılmış görüntü gösterilir.
        cv2.imshow("Foreground Mask", fgmask)
        
        # Çıkış için 'q' tuşuna basılması bekleniyor
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

vid.release()

# Tüm pencereler kapatılıyor
cv2.destroyAllWindows()
