import cv2
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


def detect_and_mesh(image_path):
    # === 入力読み込み ===
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"画像を開けません: {image_path}")
    h, w = image.shape[:2]

    # === 顔検出 ===
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            print("顔が検出されませんでした。")
            return image, None, None

        # 最初の顔のみ利用（複数顔対応も可能）
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        # バウンディングボックスを安全にクリップ
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = image[y1:y2, x1:x2].copy()

    # === FaceMeshを検出顔領域にのみ適用 ===
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as mesh:
        mesh_results = mesh.process(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        if not mesh_results.multi_face_landmarks:
            print("FaceMesh検出に失敗しました。")
            return image, None, None

        # 座標を元画像スケールに変換
        landmarks = np.array([
            [lm.x * (x2 - x1) + x1, lm.y * (y2 - y1) + y1]
            for lm in mesh_results.multi_face_landmarks[0].landmark
        ])

        # 表示例
        output = image.copy()
        for (px, py) in landmarks:
            cv2.circle(output, (int(px), int(py)), 1, (0, 255, 0), -1)

        return output, landmarks, (x1, y1, x2, y2)


if __name__ == "__main__":
    img_path = "test.png"
    output, landmarks, bbox = detect_and_mesh(img_path)

    if landmarks is not None:
        cv2.imshow("FaceMesh on Detected Face", output)
        cv2.waitKey(0)