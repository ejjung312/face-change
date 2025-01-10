import cv2
import mediapipe as mp
import numpy as np

def draw_landmarks_on_image(image, results):
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h,w,_ = image.shape
            landmarks = [
                (int(landmark.x * w), int(landmark.y * h))
                for landmark in face_landmarks.landmark
            ]
            
            image = change_lip_color(image, landmarks)
            
            # 얼굴 그물망
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_tesselation_style())
            
            # # 얼굴 윤곽 (+눈썹, 눈)
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_contours_style())
            
            # # 눈동자
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_iris_connections_style())
    
    return image


def change_lip_color(image, face_landmarks, color=(0,0,255), alpha=0.5):
    overlay = image.copy()
    # lip_points = np.array([[landmarks[idx][0], landmarks[idx][1]] for idx in LIP_LANDMARKS], np.int32)
    points = get_lip_region(image, face_landmarks)
    cv2.fillPoly(overlay, [points], color)
    
    return cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)


def get_lip_region(image, face_landmarks):
    # 입 랜드마크
    LIP_LANDMARKS = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
        95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
    ]
    
    h,w,_ = image.shape
    landmarks = [
        (int(landmark.x * w), int(landmark.y * h))
        for landmark in face_landmarks
    ]
    points = np.array([[landmarks[idx][0], landmarks[idx][1]] for idx in LIP_LANDMARKS], np.int32)
    
    return points


def get_eye_region(image, landmarks, type):
    """눈 영역 좌표 반환"""
    # 눈 랜드마크 (왼쪽과 오른쪽 눈의 Mediapipe FaceMesh 인덱스)
    LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144, 145, 153]
    RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380, 381, 382]
    EYE_LANDMARKS = []
    
    if type == 'l': EYE_LANDMARKS = LEFT_EYE_LANDMARKS
    else: EYE_LANDMARKS = RIGHT_EYE_LANDMARKS
    
    h,w,_ = image.shape
    points = np.array([(int(landmarks[idx].x*w), int(landmarks[idx].y*h)) for idx in EYE_LANDMARKS])
    
    return points


def scale_eye(image, eye_points, scale=1.5):
    """눈 크기 변경"""
    # 눈 영역의 경계 상자 계산
    x,y,w,h = cv2.boundingRect(eye_points)
    
    # 눈 영역 추출
    eye_roi = image[y:y+h, x:x+w]
    
    # 중심 좌표 계산
    center_x, center_y = x + w // 2, y + h // 2
    
    # 크기 변경
    scaled_eye = cv2.resize(eye_roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # 마스크 생성
    mask = np.zeros_like(image)
    scaled_h, scaled_w = scaled_eye.shape[:2]
    scaled_x = center_x - scaled_w // 2
    scaled_y = center_y - scaled_h // 2
    
    mask[scaled_y:scaled_y + scaled_h, scaled_x:scaled_x+scaled_w] = scaled_eye
    
    # 이미지에 눈 영역 적용
    # seamlessClone: 한 이미지의 일부를 다른 이미지에 자연스럽게 합성할 때 사용
    image = cv2.seamlessClone(scaled_eye, image, np.ones_like(scaled_eye, dtype=np.uint8)*255, (center_x, center_y), cv2.NORMAL_CLONE)
    
    return image


def face_swap(img1, img2):
    # landmarks1 = get_landmarks(img1)
    pass
    