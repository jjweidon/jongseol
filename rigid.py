import cv2
import numpy as np

def filter_good_matches(matches):
    good_matches = []
    for m in matches:
        if m.distance < 50:  # 임계값은 실험적으로 설정할 수 있습니다.
            good_matches.append(m)
    return good_matches

def rigid_registration(fixed_image, moved_image):
    # 좌우반전
    # moved_image_3 = cv2.flip(moved_image, 1)

    # 먼저 이미지를 특징점 (keypoints)으로 변환
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, des1 = orb.detectAndCompute(fixed_image, None)
    kp2, des2 = orb.detectAndCompute(moved_image, None)
    # kp3, des3 = orb.detectAndCompute(moved_image_3, None)

    # # 키 포인트 그리기
    # fixed_kp = cv2.drawKeypoints(fixed_image, kp1, None, \
    #             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # moved_kp = cv2.drawKeypoints(moved_image, kp2, None, \
    #             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # # 특징점 이미지 저장
    # cv2.imwrite('kp_fixed.jpg', fixed_kp)
    # cv2.imwrite('kp_moved.jpg', moved_kp)

    # 특징점 간의 매칭을 수행
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # # Matcher 그리기
    # matcher = bf.match(des1, des2)
    # res = cv2.drawMatches(fixed_image, kp1, moved_image, kp2, matcher, None, \
    #                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # # cv2.imwrite('BFMatcher', res)
    # cv2.imshow('BFMatcher', res)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    good_matches2 = filter_good_matches(bf.match(des1, des2))
    # good_matches3 = filter_good_matches(bf.match(des1, des3))
    # print(len(good_matches2), len(good_matches3))
    print(f"fixed_image의 특징점: {len(kp1)}개")
    print(f"moved_image의 특징점: {len(kp2)}개")
    print(f"좋은 매칭점 수: {len(good_matches2)}개")


    # 각 good_matches 리스트의 길이 계산
    # lengths = [len(good_matches2), len(good_matches3)]

    # 가장 긴 리스트의 인덱스 찾기
    # max_length_index = lengths.index(max(lengths))
    # good_matches = [good_matches2, good_matches3][max_length_index]
    # kp = [kp2, kp3][max_length_index]
    # moved_image = [moved_image, moved_image_3][max_length_index]

    # 반전 제거
    good_matches = good_matches2
    kp = kp2

    # 좋은 매칭점을 이용하여 변환 행렬을 추정
    src_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # 추정된 변환 행렬로 이미지 변환 수행
    rows, cols = fixed_image.shape
    aligned_image = cv2.warpAffine(moved_image, M, (cols, rows))

    return aligned_image

# 사용 예시
fimg = cv2.imread('./ribs/ribs_fixed.jpg', cv2.IMREAD_GRAYSCALE) # fixed image
mimg = cv2.imread('./ribs/ribs_moved.jpg', cv2.IMREAD_GRAYSCALE) # moved image
aimg = rigid_registration(fimg, mimg)

# 결과 이미지 저장 또는 표시
cv2.imwrite('./ribs/aligned_image.jpg', aimg)

def overlay_images(image1, image2, alpha=0.7):
    # 두 이미지의 크기를 동일하게 조정
    height, width = image2.shape[:2]
    resized_image1 = cv2.resize(image1, (width, height))
    # 두 이미지를 투명도(alpha)를 적용하여 겹치기
    overlay = cv2.addWeighted(resized_image1, alpha, image2, alpha, 0)
    return overlay

# 정합 전 사진과 정합 후 사진 겹치기
overlayed_image1 = overlay_images(fimg, mimg)
overlayed_image2 = overlay_images(fimg, aimg)
# 겹친 이미지 저장
cv2.imwrite('./ribs/before_overlayed_image.jpg', overlayed_image1)
cv2.imwrite('./ribs/after_overlayed_image.jpg', overlayed_image2)