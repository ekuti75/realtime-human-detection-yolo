from ultralytics import YOLO
import cv2

# YOLO modelini yükle
model = YOLO("yolo26n.pt")

# Kamerayı aç
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    # Tahmin yap
    results = model(frame)

    person_count = 0

    # Sonuçları işle
    for r in results:
        boxes = r.boxes

        for box in boxes:

            # class id
            cls = int(box.cls[0])

            # person class = 0
            if cls == 0:

                person_count += 1

                # koordinatlar
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # kutu çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

                # etiket
                cv2.putText(frame,
                            "Person",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2)

    # toplam kişi sayısı
    cv2.putText(frame,
                f"People Count: {person_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3)

    # görüntüyü göster
    cv2.imshow("Human Counter", frame)

    # q ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()