import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh

# 눈 랜드마크 (왼쪽과 오른쪽 눈의 Mediapipe FaceMesh 인덱스)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144, 145, 153]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380, 381, 382]

def get_eye_region(landmarks, eye_indexes, image_shape):
    """눈 영역 좌표를 반환"""
    h, w, _ = image_shape
    points = np.array([(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_indexes])
    return points

def scale_eye(image, eye_points, scale=1.5):
    """눈 크기를 변경"""
    # 눈 영역의 경계 상자 계산
    x, y, w, h = cv2.boundingRect(eye_points)

    # 눈 영역만 추출
    eye_roi = image[y:y + h, x:x + w]

    # 중심 좌표 계산
    center_x, center_y = x + w // 2, y + h // 2

    # 크기 변경
    scaled_eye = cv2.resize(eye_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    # 마스크 생성
    mask = np.zeros_like(image)
    scaled_h, scaled_w = scaled_eye.shape[:2]
    scaled_x = center_x - scaled_w // 2
    scaled_y = center_y - scaled_h // 2

    mask[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x + scaled_w] = scaled_eye

    # 이미지에 눈 영역 적용
    image = cv2.seamlessClone(scaled_eye, image, np.ones_like(scaled_eye, dtype=np.uint8) * 255, (center_x, center_y), cv2.NORMAL_CLONE)

    return image

def process_frame(frame, face_mesh):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # 왼쪽과 오른쪽 눈 영역 추출
            left_eye_points = get_eye_region(landmarks, LEFT_EYE_LANDMARKS, frame.shape)
            print(left_eye_points)
            right_eye_points = get_eye_region(landmarks, RIGHT_EYE_LANDMARKS, frame.shape)

            # 눈 크기 변경
            frame = scale_eye(frame, left_eye_points)
            frame = scale_eye(frame, right_eye_points)

    return frame

# 비디오 캡처
cap = cv2.VideoCapture("data/video2.mp4")

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = process_frame(frame, face_mesh)
        cv2.imshow('Eye Size Change', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키를 눌러 종료
            break

cap.release()
cv2.destroyAllWindows()
