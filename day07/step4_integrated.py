import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

# ========== 탐지할 객체 색상 프로필 정의 ==========
# 각 객체: 이름, HSV 범위 2개, 바운딩박스 색상(BGR), 최소 면적
COLOR_PROFILES = [
    {
        "name": "Stop Sign (Red)",
        "lower1": np.array([0, 100, 100]),
        "upper1": np.array([10, 255, 255]),
        "lower2": np.array([170, 100, 100]),
        "upper2": np.array([180, 255, 255]),
        "color": (0, 0, 255),
        "min_area": 500,
        "min_vertices": 6,
    },
    {
        "name": "Warning Sign (Yellow)",
        "lower1": np.array([20, 100, 100]),
        "upper1": np.array([35, 255, 255]),
        "lower2": None,
        "upper2": None,
        "color": (0, 215, 255),
        "min_area": 500,
        "min_vertices": 3,
    },
    {
        "name": "Info Sign (Blue)",
        "lower1": np.array([100, 100, 50]),
        "upper1": np.array([130, 255, 255]),
        "lower2": None,
        "upper2": None,
        "color": (255, 100, 0),
        "min_area": 500,
        "min_vertices": 3,
    },
    {
        "name": "Go Sign (Green)",
        "lower1": np.array([40, 80, 80]),
        "upper1": np.array([80, 255, 255]),
        "lower2": None,
        "upper2": None,
        "color": (0, 200, 0),
        "min_area": 500,
        "min_vertices": 3,
    },
]

def detect_by_color(img, hsv, profile):
    """색상 프로필로 객체 탐지 후 결과 반환"""
    # 마스크 생성
    mask = cv.inRange(hsv, profile["lower1"], profile["upper1"])
    if profile["lower2"] is not None:
        mask2 = cv.inRange(hsv, profile["lower2"], profile["upper2"])
        mask = cv.bitwise_or(mask, mask2)

    # 노이즈 제거
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # 컨투어 검출
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    detected = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area < profile["min_area"]:
            continue

        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * perimeter, True)
        num_vertices = len(approx)

        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0

        if num_vertices >= profile["min_vertices"] and 0.5 <= aspect_ratio <= 2.0:
            detected.append({
                "x": x, "y": y, "w": w, "h": h,
                "vertices": num_vertices,
                "area": int(area),
                "name": profile["name"],
                "color": profile["color"],
            })

    return mask, detected


def draw_results(img, all_detected):
    """탐지 결과를 이미지에 그리기"""
    result = img.copy()
    for obj in all_detected:
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        cv.rectangle(result, (x, y), (x+w, y+h), obj["color"], 3)
        label = f'{obj["name"]} ({obj["vertices"]}v)'
        cv.putText(result, label, (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, obj["color"], 2)
    return result


def process_image(img_path):
    """단일 이미지 처리"""
    img = cv.imread(img_path)
    if img is None:
        print(f"이미지 로드 실패: {img_path}")
        return

    print(f"\n이미지: {img_path} {img.shape}")
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    start = time.time()
    all_detected = []
    all_masks = []

    for profile in COLOR_PROFILES:
        mask, detected = detect_by_color(img, hsv, profile)
        all_masks.append((profile["name"], mask))
        all_detected.extend(detected)
        if detected:
            print(f"  ✓ {profile['name']}: {len(detected)}개 검출 "
                  f"(vertices={[d['vertices'] for d in detected]}, "
                  f"area={[d['area'] for d in detected]})")

    elapsed = (time.time() - start) * 1000
    print(f"  처리 시간: {elapsed:.1f}ms | 총 검출: {len(all_detected)}개")

    result_img = draw_results(img, all_detected)

    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Integrated Color Detection — {len(all_detected)} objects found', fontsize=14)

    axes[0, 0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv.cvtColor(result_img, cv.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Detection Result ({len(all_detected)} objects)')
    axes[0, 1].axis('off')

    for i, (name, mask) in enumerate(all_masks[:4]):
        row, col = divmod(i + 2, 3)
        if row < 2 and col < 3:
            axes[row, col].imshow(mask, cmap='gray')
            axes[row, col].set_title(f'Mask: {name.split("(")[1].rstrip(")")}')
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('step4_result.png', dpi=150)
    plt.close()
    print("  결과 저장: step4_result.png")


def process_webcam():
    """웹캠 실시간 처리 (선택)"""
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("웹캠 실행 중... 'q' 누르면 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        all_detected = []
        for profile in COLOR_PROFILES:
            _, detected = detect_by_color(frame, hsv, profile)
            all_detected.extend(detected)

        result = draw_results(frame, all_detected)
        cv.putText(result, f'Objects: {len(all_detected)}', (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.imshow('Integrated Detection (q: quit)', result)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


# ========== 메인 실행 ==========
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "webcam":
            process_webcam()
        else:
            # 여러 이미지 동시 처리 가능
            for img_path in sys.argv[1:]:
                process_image(img_path)
    else:
        # 기본: stop_sign.jpg 사용
        print("사용법: python step4_integrated.py <이미지파일> [이미지파일2 ...]")
        print("        python step4_integrated.py webcam")
        print("\n기본 이미지(stop_sign.jpg)로 실행합니다...")
        process_image("stop_sign.jpg")
