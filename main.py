import cv2
import mediapipe as mp
import util as util

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

webCam = True
video_path = "data/video2.mp4"

cap = cv2.VideoCapture(video_path)
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    
    while True:
        if webCam:
            success, image = cap.read()
        else:
            image = cv2.imread("data/1.png")

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw the face mesh annotations on the image.
        # image = util.draw_landmarks_on_image(image, results)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # change lip color
                image = util.change_lip_color(image, landmarks, color=(3,3,210))
                
                # get left/right eye region
                left_eye_points = util.get_eye_region(image, landmarks, 'l')
                right_eye_points = util.get_eye_region(image, landmarks, 'r')
                
                # scale eyes
                image = util.scale_eye(image, left_eye_points, scale=1.2)
                image = util.scale_eye(image, right_eye_points, scale=1.2)
        
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Face Change', cv2.flip(image, 1))
        
        if cv2.waitKey(50) & 0xFF == 27:
            break
        
cap.release()