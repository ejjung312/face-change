import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def get_face_landmarks(image):
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0]
    return None

def warp_face(src_img, src_points, dst_img, dst_points):
    hull_index = cv2.convexHull(np.array(dst_points), returnPoints=False)
    hull_src = [src_points[int(idx)] for idx in hull_index]
    hull_dst = [dst_points[int(idx)] for idx in hull_index]
    
    mask = np.zeros_like(dst_img, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(hull_dst), (255, 255, 255))
    rect = cv2.boundingRect(np.float32(hull_dst))
    
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(hull_dst)
    triangles = subdiv.getTriangleList()
    triangles = np.array([t.reshape(3, 2) for t in triangles], dtype=np.float32)

    for t in triangles:
        pts_src = [src_points[dst_points.index(tuple(pt))] for pt in t]
        pts_dst = t
        warp_mat = cv2.getAffineTransform(np.float32(pts_src), np.float32(pts_dst))
        warped = cv2.warpAffine(src_img, warp_mat, (dst_img.shape[1], dst_img.shape[0]))
        mask_tri = np.zeros_like(dst_img, dtype=np.uint8)
        cv2.fillConvexPoly(mask_tri, np.int32(pts_dst), (255, 255, 255))
        dst_img = cv2.bitwise_and(dst_img, cv2.bitwise_not(mask_tri))
        dst_img = cv2.add(dst_img, warped)
    
    return dst_img

def extract_points(landmarks, img):
    h, w = img.shape[:2]
    return [(int(pt.x * w), int(pt.y * h)) for pt in landmarks.landmark]

def blend_faces(img1, img2, warped_face, mask):
    center = (mask.shape[1] // 2, mask.shape[0] // 2)
    return cv2.seamlessClone(warped_face, img1, mask, center, cv2.NORMAL_CLONE)

# 입력 이미지
image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')

# 얼굴 특징점 추출
landmarks1 = get_face_landmarks(image1)
landmarks2 = get_face_landmarks(image2)

if landmarks1 and landmarks2:
    points1 = extract_points(landmarks1, image1)
    points2 = extract_points(landmarks2, image2)
    
    # 이미지 2 얼굴을 이미지 1에 매핑
    warped_face = warp_face(image2, points2, image1, points1)
    
    # 얼굴 합성
    mask = np.zeros_like(image1, dtype=np.uint8)
    hull = cv2.convexHull(np.array(points1, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    
    output = blend_faces(image1, image2, warped_face, mask)
    cv2.imshow('Swapped Face', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("얼굴을 감지할 수 없습니다.")
