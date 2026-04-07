import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sample_download import get_sample

# ========== Step 1: 이미지 로드 ==========
img1 = cv.imread(get_sample('box.png'), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(get_sample('box_in_scene.png'), cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: 이미지를 찾을 수 없습니다.")
    exit()

print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}")

# ========== Step 2: 특징점 검출기 초기화 ==========
# SIFT 사용 (정확도 우선)
USE_SIFT = True  # False로 바꾸면 ORB 사용

if USE_SIFT:
    detector = cv.SIFT_create()
    print("사용 중: SIFT")
else:
    detector = cv.ORB_create()
    print("사용 중: ORB")

# ========== Step 3: 키포인트와 디스크립터 추출 ==========
kp1, des1 = detector.detectAndCompute(img1, None)
kp2, des2 = detector.detectAndCompute(img2, None)

print(f"Keypoints found - img1: {len(kp1)}, img2: {len(kp2)}")

# ========== Step 4: FLANN 매칭기 설정 ==========
if USE_SIFT:
    # SIFT: float descriptor → KDTREE
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
else:
    # ORB: binary descriptor → LSH
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=12, key_size=20, multi_probe_level=2)
    search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)

# ========== Step 5: knnMatch로 k=2 매칭 ==========
matches = flann.knnMatch(des1, des2, k=2)
print(f"Total matches: {len(matches)}")

# ========== Step 6: Lowe's 비율 테스트 ==========
RATIO = 0.7  # 여기 값 바꿔서 실험해봐 (0.5 ~ 0.9)

good_matches = []
for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < RATIO * n.distance:
            good_matches.append(m)

print(f"Good matches after Lowe's ratio test (ratio={RATIO}): {len(good_matches)}")

# ========== Step 7: 시각화 ==========
if len(good_matches) >= 10:
    result_img = cv.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(16, 6))
    plt.imshow(result_img, cmap='gray')
    plt.title(f"Good Matches ({len(good_matches)}) — Lowe's ratio={RATIO}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('step1_result.png', dpi=150)
    plt.show()
    print("결과 저장됨: step1_result.png")
else:
    print(f"매칭 개수 부족: {len(good_matches)}개 (최소 10개 필요)")
