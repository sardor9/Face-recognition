import cv2

def detect_faces_and_eyes():
    # Загружаем встроенные каскады из OpenCV
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    # Проверяем загрузку
    if face_cascade.empty() or eye_cascade.empty():
        raise IOError("Ошибка загрузки каскадов Хаара")

    # Читаем изображение или с камеры
    cap = cv2.VideoCapture(0)  # 0 = встроенная вебка
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Находим лица
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Для каждого лица ищем глаза
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        cv2.imshow("Face and Eye Detection", frame)

        # Выход по клавише Esc
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Запуск
detect_faces_and_eyes()
