import os
import cv2
import dlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider


# WIDER FACE 데이터셋 경로 (사용자의 저장 경로 사용)
WIDER_FACE_DIR = r"C:\Users\User\Desktop\Lips_Landmark"
IMAGE_DIR = os.path.join(WIDER_FACE_DIR, "WIDER_train", "images")
ANNOTATION_FILE = os.path.join(WIDER_FACE_DIR, "wider_face_split", "wider_face_train_bbx_gt.txt")

# dlib 얼굴 랜드마크 모델 로드
PREDICTOR_PATH = r"C:\Users\User\Desktop\Lips_Landmark\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# 바운딩 박스 데이터를 불러오는 함수
def load_widerface_annotations(annotation_path):
    annotations = {}
    with open(annotation_path, "r") as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            image_name = lines[i].strip()
            i += 1
            if i >= len(lines):
                break
            
            try:
                num_faces = int(lines[i].strip())  # 얼굴 개수
                i += 1
                bboxes = []
                for j in range(num_faces):
                    bbox_data = list(map(int, lines[i].strip().split()[:4]))  # x, y, w, h만 추출
                    bboxes.append(bbox_data)
                    i += 1
                annotations[image_name] = bboxes
            except ValueError:
                print(f"Warning: Skipping annotation for {image_name}")
                continue
    return annotations

# dlib을 사용하여 랜드마크를 추출하는 함수
def extract_landmarks(image_path, bboxes):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks_list = []
    
    for bbox in bboxes:
        x, y, w, h = bbox
        rect = dlib.rectangle(x, y, x + w, y + h)
        shape = predictor(gray, rect)
        
        # 필요한 5개 랜드마크 좌표 추출
        keypoints = [
            (shape.part(36).x, shape.part(36).y),  # 왼쪽 눈 (눈 시작점)
            (shape.part(45).x, shape.part(45).y),  # 오른쪽 눈 (눈 끝점)
            (shape.part(30).x, shape.part(30).y),  # 코 끝점
            (shape.part(48).x, shape.part(48).y),  # 왼쪽 입꼬리
            (shape.part(54).x, shape.part(54).y)   # 오른쪽 입꼬리
        ]
        landmarks_list.append(keypoints)
    
    return landmarks_list

# 바운딩 박스 + 랜드마크 시각화 함수 (Matplotlib 방식으로 수정)
def visualize_landmarks(image_path, bboxes, landmarks):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2)
    ax.imshow(image)
    ax.axis("off")
    
    # 바운딩 박스 추가
    for bbox in bboxes:
        x, y, w, h = bbox
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    
    # 랜드마크 추가
    for landmark in landmarks:
        landmark_x = [pt[0] for pt in landmark]
        landmark_y = [pt[1] for pt in landmark]
        ax.scatter(landmark_x, landmark_y, c='lime', marker='o', s=20)
    
    # 확대/축소 슬라이더 추가
    ax_zoom = plt.axes([0.2, 0.05, 0.65, 0.03])
    zoom_slider = Slider(ax_zoom, 'Zoom', 1.0, 3.0, valinit=1.0)
    
    def update(val):
        zoom = zoom_slider.val
        ax.set_xlim([image.shape[1] / 2 - (image.shape[1] / (2 * zoom)),
                     image.shape[1] / 2 + (image.shape[1] / (2 * zoom))])
        ax.set_ylim([image.shape[0] / 2 + (image.shape[0] / (2 * zoom)),
                     image.shape[0] / 2 - (image.shape[0] / (2 * zoom))])
        fig.canvas.draw_idle()
    
    zoom_slider.on_changed(update)
    plt.show()

# 바운딩 박스 및 랜드마크 추출 및 시각화
annotations = load_widerface_annotations(ANNOTATION_FILE)

# 샘플 이미지 선택 (최대 3개)
sample_images = list(annotations.keys())[:10]
for sample_image in sample_images:
    sample_image_path = os.path.join(IMAGE_DIR, sample_image.replace("/", os.sep))
    if os.path.exists(sample_image_path):
        sample_bboxes = annotations[sample_image]
        sample_landmarks = extract_landmarks(sample_image_path, sample_bboxes)
        print(f"Visualizing: {sample_image}")
        visualize_landmarks(sample_image_path, sample_bboxes, sample_landmarks)
    else:
        print(f"Error: Image file not found - {sample_image_path}")