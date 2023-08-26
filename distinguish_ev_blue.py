import os # 운영체제와 상호작용하기 위함
import glob  # 특정 패턴에 맞는 경로명 찾기 위함
import cv2 # 이미지 처리 및 cv 작업 위함
from ultralytics import YOLO # yolo 모델 가져오기
import numpy as np

# 훈련된 yolo 모델의 가중치를 로드해서 객체 탐지 모델 인스턴스 생성
model = YOLO("/Users/dobby/Library/Mobile Documents/com~apple~CloudDocs/electronicsandcode/code/arcticfo111/ms_06team/car_plate_detection_yolov8/runs/detect/label_class4_multi_car_all/best.pt")

# 테스트 데이터가 있는 폴더 경로 찾기
data_path = "/Users/dobby/Library/Mobile Documents/com~apple~CloudDocs/electronicsandcode/code/arcticfo111/ms_06team/ms_team_cardata/test_real_car/20230822_162037.mp4"

# 최종 결과값 저장할 디렉토리
save_path_result = "./result"
if not os.path.exists(save_path_result):
    os.mkdir(save_path_result)

# 번호판은 탐지 안 되고, 차만 탐지 된 경우 결과값 저장할 디렉토리
save_path_detection_error = "./detection_error"
if not os.path.exists(save_path_detection_error):
    os.mkdir(save_path_detection_error)

# 파랑 번호판이 아닌 이미지를 저장할 디렉토리
save_path_non_color = "./non_color"
if not os.path.exists(save_path_non_color):
    os.mkdir(save_path_non_color)

# 파랑 번호판이미지를 저장할 디렉토리
save_path_blue = "./blue_detected"
if not os.path.exists(save_path_blue):
    os.mkdir(save_path_blue)

# 두줄 번호판이미지를 저장할 디렉토리
save_path_twoline = "./twoline_detected"
if not os.path.exists(save_path_twoline):
    os.mkdir(save_path_twoline)


# 파일의 확장자 가져오기
file_extension = os.path.splitext(data_path)[1]

# 일반적인 비디오 파일 형식 리스트업
video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".MOV"]

# 제공한 파일이 비디오 형식이라면
if file_extension in video_extensions:
    # 비디오 프레임 저장할 경로
    save_dir = './video_frames'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # 비디오 캡쳐 객체 생성
    cap = cv2.VideoCapture(data_path)
    
    # 비디오의 FPS frames per second 값 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 1초마다 프레임 저장
        if frame_count % fps == 0:
            save_path = os.path.join(save_dir, f'frame_{frame_count}.png')
            cv2.imwrite(save_path, frame)
            print(f"Saved: frame_{frame_count}.png")

        frame_count += 1

    cap.release()
    
    data_path = save_dir


# 테스트 이미지가 있는 폴더에서 모든 이미지 파일의 경로를 찾아서 리스트에 저장하기
data_path_list = glob.glob(os.path.join(data_path, "*.png")) 

# 각각의 이미지 파일에 대해 for 문
for path in data_path_list :
    # 테스트 이미지가 있는  경로에서 이미지 하나씩 읽기
    image = cv2.imread(path)

    # 클래스 목록 가져오기
    names = model.names

    '''
    이미지에서 객체를 탐지하고, 결과 반환하기 
    이미지파일경로, 이미지 파일로 자동저장할지 여부, 욜로 입력 데이터 크기를 고혀해 리사이징 할 이미지크기, 신뢰도 임계값(예를들어 0.7은 욜로가 예측한 각 객체의 신뢰도가 70% 이상일 때만 유효한 탐지로 간주한다)
    '''
    results = model.predict(path, save=False, imgsz=640, conf=0.7)
    
    # 결과에서 바운딩 박스 정보 가져오기
    boxes = results[0].boxes
    results_info = boxes
    
    # 한 이미지에 대해 번호판 색 구성을 저장하는 리스트
    plate_color = []
    
    # 각 바운딩 박스의 클래스 번호, 바운딩 박스 신뢰도, 바운딩 박스 좌표 점수 가져오기
    cls_numbers = results_info.cls 
    conf_numbers = results_info.conf 
    box_xyxy = results_info.xyxy
    # 각 바운딩 박스 클래스 이름, 인덱스 담을 변수 초기화
    class_number, class_name = None, None
    # 각 바운딩 박스의 클래스 번호, 바운딩 박스 신뢰도, 바운딩 박스 좌표에 대해 for 문
    for bbox, cls_idx, conf_idx in zip(box_xyxy, cls_numbers, conf_numbers) :
        class_number = int(cls_idx.item()) # 클래스 인덱스를 정수로 변환
        class_name = names[class_number] # 클래스 이름을 가져오기
        
        # 바운딩 박스의 왼쪽, 오른족 x, y 값 각각 가져오기
        x1 = int(bbox[0].item()) 
        y1 = int(bbox[1].item())
        x2 = int(bbox[2].item())
        y2 = int(bbox[3].item())
        
        # 이미지에 녹색 바운딩 박스 그리기
        rect = cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        
        '''
            번호판 색 탐지하기
            클래스가 차가 아닌 경우(번호판인 경우)에 실행
        '''
        if class_name != 'car': 

            # 번호판 영역 추출
            plate = image[y1:y2, x1:x2]

            if class_name == 'ev_plate':
                filename = os.path.basename(path)
                cv2.imwrite(os.path.join(save_path_blue, filename), plate)              
            elif class_name == 'two_line_plate':
                filename = os.path.basename(path)
                cv2.imwrite(os.path.join(save_path_twoline, filename), plate)
            elif class_name == 'one_line_plate':            
                # HSV로 변환
                hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
            
                # 파란색 범위 정의
                lower_blue = np.array([95, 50, 50]) # 색상, 채도, 명도
                upper_blue = np.array([140, 255, 255])
                mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            
                # 파란색 픽셀 비율 계산
                blue_ratio = np.sum(mask_blue) / (mask_blue.shape[0] * mask_blue.shape[1] * 255)
            
                # 노란색 범위 정의
                lower_yellow = np.array([20, 100, 100])
                upper_yellow = np.array([50, 255, 255])

                # 노란색 픽셀 비율 계산
                mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
                yellow_ratio = np.sum(mask_yellow) / (mask_yellow.shape[0] * mask_yellow.shape[1] * 255)

                # 초록색 범위 정의
                lower_green = np.array([35, 50, 50])
                upper_green = np.array([85, 255, 255])
            
                # 초록색 픽셀 비율 계산
                mask_green = cv2.inRange(hsv, lower_green, upper_green)
                green_ratio = np.sum(mask_green) / (mask_green.shape[0] * mask_green.shape[1] * 255)

                # 번호판 색 여부에 따라 이미지 속 번호판 색을 저장하는 리스트에 요소 추가
                if blue_ratio >= 0.65:                
                    plate_color.append('blue')  
                    filename = os.path.basename(path)
                    cv2.imwrite(os.path.join(save_path_blue, filename), plate)              
                elif yellow_ratio >= 0.65:  # 노란색이 이미지의 50% 이상일 경우
                    plate_color.append('yellow')
                    filename = os.path.basename(path)
                    cv2.imwrite(os.path.join(save_path_twoline, filename), plate)
                elif green_ratio >= 0.5:  # 노란색이 이미지의 50% 이상일 경우
                    plate_color.append('green')
                    filename = os.path.basename(path)
                    cv2.imwrite(os.path.join(save_path_twoline, filename), plate)
                else:
                    plate_color.append('white')
                    filename = os.path.basename(path)
                    cv2.imwrite(os.path.join(save_path_non_color, filename), plate)

    # 번호판이 탐지 되지 않은 경우, detection_error 결과 폴더로 결과를 저장하기
    if not((cls_numbers == 0.).any().item() and any((cls_numbers == x).any().item() for x in [1., 2., 3.])):     
        print(plate_color, cls_numbers)
        filename = os.path.basename(path)
        cv2.imwrite(os.path.join(save_path_detection_error, filename), image)
    # 전체 결과 저장하기
    filename = os.path.basename(path)
    cv2.imwrite(os.path.join(save_path_result, filename), image)