import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sample_download import get_sample

# ========== 이미지 로드 ==========
img1 = cv.imread(get_sample('box.png'), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(get_sample('box_in_scene.png'), cv.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("Error: 이미지를 찾을 수 없습니다.")
    exit()

# ========== SIFT 특징점 추출 ==========
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# ========== FLANN 매칭 ==========
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

print(f"Good matches: {len(good_matches)}")

# ========== Step 1: 호모그래피 계산 ==========
MIN_MATCH_COUNT = 10

if len(good_matches) >= MIN_MATCH_COUNT:
    # good_matches에서 키포인트 좌표 추출
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 호모그래피 행렬 계산 (RANSAC)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

    if M is not None:
        # img1의 네 모서리 좌표 정의
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        # perspectiveTransform으로 img2에서의 위치 계산
        dst = cv.perspectiveTransform(pts, M)

        # ========== Step 2: 결과 시각화 ==========
        # img2가 grayscale이면 BGR로 변환해서 색깔 선 그리기
        result_img = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
        cv.polylines(result_img, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)

        plt.figure(figsize=(10, 8))
        plt.imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
        plt.title('Detected Object with Homography')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('step2_homography.png', dpi=150)
        plt.close()

        # ========== Step 3: inlier 매칭만 표시 ==========
        matchesMask = mask.ravel().tolist()

        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=matchesMask,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

        plt.figure(figsize=(16, 6))
        plt.imshow(img_matches, cmap='gray')
        plt.title('Inlier Matches Only')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('step2_inliers.png', dpi=150)
        plt.close()

        inlier_count = sum(matchesMask)
        outlier_count = len(matchesMask) - inlier_count
        print(f"Inliers: {inlier_count}, Outliers: {outlier_count}")
        print("결과 저장됨: step2_homography.png, step2_inliers.png")

    else:
        print("호모그래피 계산 실패 (M is None)")

else:
    print(f"매칭 개수 부족 ({len(good_matches)}/{MIN_MATCH_COUNT})")
