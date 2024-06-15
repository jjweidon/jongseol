# rigid.py

import cv2
import numpy as np

def calculate_similarity_percentage(kp1, kp2, threshold_distance=10):
    similar_points = 0
    total_points = min(len(kp1), len(kp2))

    for point1 in kp2:
        for point2 in kp1:
            distance = np.linalg.norm(np.array(point1.pt) - np.array(point2.pt))
            if distance < threshold_distance:
                similar_points += 1
                break

    similarity_percentage = (similar_points / total_points) * 100
    return similarity_percentage

def overlay_images(image1, image2, alpha=0.7):
    # 두 이미지의 크기를 동일하게 조정
    height, width = image2.shape[:2]
    resized_image1 = cv2.resize(image1, (width, height))

    # 두 이미지를 투명도(alpha)를 적용하여 겹치기
    overlay = cv2.addWeighted(resized_image1, alpha, image2, alpha, 0)
    return overlay

def get_images():
    # content = 'lung3' # 1100, 100
    content = 'brain2' # 1600, 50

    fimg = cv2.imread(f'./img/{content}/{content}_fixed.jpg', cv2.IMREAD_GRAYSCALE) # fixed image
    mimg = cv2.imread(f'./img/{content}/{content}_moved.jpg', cv2.IMREAD_GRAYSCALE) # moved image

    orb = cv2.ORB_create(nfeatures=1600)
    kp1, ds1 = orb.detectAndCompute(fimg, None)
    kp2, ds2 = orb.detectAndCompute(mimg, None)

    # 키 포인트 그리기
    fixed_kp = cv2.drawKeypoints(fimg, kp1, None, \
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    moved_kp = cv2.drawKeypoints(mimg, kp2, None, \
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # 특징점 이미지 저장
    cv2.imwrite(f'./img/{content}/kp_fixed.jpg', fixed_kp)
    cv2.imwrite(f'./img/{content}/kp_moved.jpg', moved_kp)

    # 특징점 간의 매칭을 수행
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    good_matches = []
    imgye = 50 # distance 임계값
    for m in bf.match(ds1, ds2):
        if m.distance < imgye:  # 임계값은 실험적으로 설정할 수 있습니다.
            good_matches.append(m)

    print(f"fixed_image의 특징점: {len(kp1)}개 / 특징점 1개당 디스크립터: {ds1.shape[1]}개")
    print(f"moved_image의 특징점: {len(kp2)}개 / 특징점 1개당 디스크립터: {ds2.shape[1]}개")
    print(f"좋은 매칭점 수: {len(good_matches)}개")

    # 좋은 매칭점을 이용하여 변환 행렬을 추정
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # 추정된 변환 행렬로 이미지 변환 수행
    rows, cols = fimg.shape
    aimg = cv2.warpAffine(mimg, M, (cols, rows))
    cv2.imwrite(f'./img/{content}/{content}_aligned.jpg', aimg)

    # 유사도 검사
    kp1, des1 = orb.detectAndCompute(fimg, None)
    kp2, des2 = orb.detectAndCompute(mimg, None)
    kp4, des4 = orb.detectAndCompute(aimg, None)
    similarity_percentage1 = calculate_similarity_percentage(kp1, kp2)
    print(f"정합 전 두 이미지의 키 포인트 유사도: {similarity_percentage1:.2f}%")
    similarity_percentage2 = calculate_similarity_percentage(kp1, kp4)
    print(f"정합 후 두 이미지의 키 포인트 유사도: {similarity_percentage2:.2f}%")

    # 정합 전 사진과 정합 후 사진 겹치기
    overlayed_image1 = overlay_images(fimg, mimg)
    overlayed_image2 = overlay_images(fimg, aimg)
    # 겹친 이미지 저장
    cv2.imwrite(f'./img/{content}/before_overlayed_image.jpg', overlayed_image1)
    cv2.imwrite(f'./img/{content}/after_overlayed_image.jpg', overlayed_image2)

    return content, fimg, mimg, aimg

if __name__ == "__main__":
    get_images()
