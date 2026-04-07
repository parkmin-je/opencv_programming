import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

# ========== Step 1: 정지 표지판 이미지 로드 ==========
# 실행 방법: python step3_stop_sign.py stop_sign.jpg
# 파일명 인자 없으면 'stop_sign.jpg' 기본값 사용

img_path = sys.argv[1] if len(sys.argv) > 1 else 'stop_sign.jpg'
img = cv.imread(img_path)

if img is None:
    print(f"이미지를 찾을 수 없습니다: {img_path}")
    print("사용법: python step3_stop_sign.py <이미지파일>")
    exit()

print(f"이미지 로드 성공: {img.shape}")

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# ========== Step 2: 빨간색 마스크 생성 ==========
# HSV에서 빨강은 두 구간으로 나뉨: 0~10, 170~180
# 필요하면 이 값 조정해봐

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask1 = cv.inRange(hsv, lower_red1, upper_red1)
mask2 = cv.inRange(hsv, lower_red2, upper_red2)
red_mask = cv.bitwise_or(mask1, mask2)

print(f"Red pixels (before morphology): {cv.countNonZero(red_mask)}")

# ========== Step 3: 노이즈 제거 (모폴로지) ==========
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)  # 구멍 채우기
red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)   # 작은 노이즈 제거

print(f"Red pixels (after morphology): {cv.countNonZero(red_mask)}")

# ========== Step 4: 컨투어 검출 ==========
contours, _ = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print(f"Found {len(contours)} contours")

# ========== Step 5: 컨투어 필터링 ==========
min_area = 500  # 너무 작은 노이즈 제거 (이미지 크기에 따라 조정)

detected_signs = []

for contour in contours:
    area = cv.contourArea(contour)
    if area < min_area:
        continue

    # 근사 다각형 계산
    perimeter = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
    num_vertices = len(approx)

    # 바운딩 박스 및 aspect ratio
    x, y, bw, bh = cv.boundingRect(contour)
    aspect_ratio = float(bw) / bh if bh > 0 else 0

    # 6각형 이상 + 정사각형에 가까운 형태 (정지 표지판 = 팔각형)
    if num_vertices >= 6 and 0.8 <= aspect_ratio <= 1.2:
        detected_signs.append((x, y, bw, bh, num_vertices, area))

print(f"Detected stop signs: {len(detected_signs)}")

# ========== Step 6: 결과 시각화 ==========
result_img = img.copy()

for x, y, bw, bh, vertices, area in detected_signs:
    cv.rectangle(result_img, (x, y), (x+bw, y+bh), (0, 0, 255), 3)
    label = f'Stop ({vertices}v, area={int(area)})'
    cv.putText(result_img, label, (x, y-10),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

plt.figure(figsize=(14, 6))

plt.subplot(131)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')

plt.subplot(132)
plt.imshow(red_mask, cmap='gray')
plt.title('Red Color Mask')
plt.axis('off')

plt.subplot(133)
plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
plt.title(f'Detected Stop Signs ({len(detected_signs)})')
plt.axis('off')

plt.tight_layout()
plt.savefig('step3_result.png', dpi=150)
plt.show()
print("결과 저장됨: step3_result.png")
