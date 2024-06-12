# nonrigid.py

import SimpleITK as sitk
import cv2
import numpy as np
import time
from rigid import overlay_images, get_images, calculate_similarity_percentage
from skimage.metrics import structural_similarity as ssim

def command_iteration(method):
    print(f"최적화 반복: {method.GetOptimizerIteration()}, 메트릭 값: {method.GetMetricValue()}")

def b_spline_registration():
    start_time = time.time()
    grid = 30.0
    learning_rate = 0.5
    number_of_iterations = 200

    # 1. 이미지 불러오기
    step_start_time = time.time()
    print("1단계: 이미지 불러오기")
    content, fimg, aimg = get_images()
    fixed_image_cv = fimg
    moved_image_cv = aimg
    step_end_time = time.time()
    print(f"1단계 완료: {step_end_time - step_start_time:.2f} 초 소요")

    # 2. OpenCV 이미지를 SimpleITK 이미지로 변환
    step_start_time = time.time()
    print("2단계: 이미지를 SimpleITK 포맷으로 변환")
    fixed_image = sitk.GetImageFromArray(fixed_image_cv.astype(np.float32))
    moved_image = sitk.GetImageFromArray(moved_image_cv.astype(np.float32))
    step_end_time = time.time()
    print(f"2단계 완료: {step_end_time - step_start_time:.2f} 초 소요")

    # 3. B-spline 그리드 초기화
    step_start_time = time.time()
    print("3단계: B-spline 그리드 초기화")
    grid_physical_spacing = [grid, grid]  # 각 그리드 포인트 간격 설정
    image_physical_size = [fixed_image.GetSize()[i] * fixed_image.GetSpacing()[i] for i in range(2)]
    mesh_size = [int(image_physical_size[i] / grid_physical_spacing[i] + 0.5) for i in range(2)]
    initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image, transformDomainMeshSize=mesh_size, order=3)
    step_end_time = time.time()
    print(f"3단계 완료: {step_end_time - step_start_time:.2f} 초 소요")

    # 4. 정합 방법 설정
    step_start_time = time.time()
    print("4단계: 정합 방법 설정")
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=learning_rate, 
                                                                numberOfIterations=number_of_iterations, 
                                                                convergenceMinimumValue=1e-6, 
                                                                convergenceWindowSize=10)
    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    registration_method.SetInterpolator(sitk.sitkLinear)

    print(f'gird: {grid}, learning_rate: {learning_rate}, number_of_iterations: {number_of_iterations}')
    # 콜백 함수 설정
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(registration_method))
    step_end_time = time.time()
    print(f"4단계 완료: {step_end_time - step_start_time:.2f} 초 소요")

    # 5. B-spline registration 수행
    step_start_time = time.time()
    print("5단계: B-spline 정합 수행")
    final_transform = registration_method.Execute(fixed_image, moved_image)
    step_end_time = time.time()
    print(f"5단계 완료: {step_end_time - step_start_time:.2f} 초 소요")

    # 경과 시간 출력
    elapsed_time = time.time() - start_time
    print(f"총 경과 시간: {elapsed_time:.2f} 초")

    # 6. 변환된 이미지를 적용하여 정렬된 결과 이미지 생성
    step_start_time = time.time()
    print("6단계: 결과 이미지 재샘플링 및 저장")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)

    out = resampler.Execute(moved_image)
    out_array = sitk.GetArrayFromImage(out).astype(np.uint8)

    # 결과 이미지 저장
    cv2.imwrite(f'./img/{content}/final_aligned_image.jpg', out_array)
    step_end_time = time.time()
    print(f"6단계 완료: {step_end_time - step_start_time:.2f} 초 소요")
    print(f"결과 이미지가 저장되었습니다.")

    overlayed_image = overlay_images(fixed_image_cv, out_array)
    cv2.imwrite(f'./img/{content}/nonrigid_overlayed_image.jpg', overlayed_image)

    # 정합 후 유사도 계산
    rigid_ssim = ssim(fixed_image_cv, moved_image_cv) * 100
    final_ssim = ssim(fixed_image_cv, out_array) * 100  
    print(f"정합 전 SSIM: {rigid_ssim:.2f}%")
    print(f"정합 후 SSIM: {final_ssim:.2f}%")

    # 키포인트 정확도
    orb = cv2.ORB_create(nfeatures=1100)
    kp1, des1 = orb.detectAndCompute(fimg, None)
    kp2, des2 = orb.detectAndCompute(aimg, None)
    kp4, des4 = orb.detectAndCompute(out_array, None)
    similarity_percentage1 = calculate_similarity_percentage(kp1, kp2)
    print(f"정합 전 두 이미지의 키 포인트 유사도: {similarity_percentage1:.2f}%")
    similarity_percentage2 = calculate_similarity_percentage(kp1, kp4)
    print(f"정합 후 두 이미지의 키 포인트 유사도: {similarity_percentage2:.2f}%")

if __name__ == "__main__":
    b_spline_registration()