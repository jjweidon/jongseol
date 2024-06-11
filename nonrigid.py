import SimpleITK as sitk
import cv2
import numpy as np
import time
from rigid_ORB import overlay_images
from skimage.metrics import structural_similarity as ssim

def command_iteration(method):
    print(f"최적화 반복: {method.GetOptimizerIteration()}, 메트릭 값: {method.GetMetricValue()}")

def preprocess_image(image_path):
    # 이미지를 읽고 전처리하는 함수
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def b_spline_registration(fixed_image_path, moved_image_path, output_image_path):
    start_time = time.time()

    # 1. 이미지 전처리
    step_start_time = time.time()
    print("1단계: 이미지 전처리")
    fixed_image_cv = preprocess_image(fixed_image_path)
    moved_image_cv = preprocess_image(moved_image_path)
    step_end_time = time.time()
    print(f"1단계 완료: {step_end_time - step_start_time:.2f} 초 소요")

    # 2. OpenCV 이미지를 SimpleITK 이미지로 변환
    step_start_time = time.time()
    print("2단계: 이미지를 SimpleITK 포맷으로 변환")
    # fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    # moved_image = sitk.ReadImage(moved_image_path, sitk.sitkFloat32)
    fixed_image = sitk.GetImageFromArray(fixed_image_cv.astype(np.float32))
    moved_image = sitk.GetImageFromArray(moved_image_cv.astype(np.float32))
    step_end_time = time.time()
    print(f"2단계 완료: {step_end_time - step_start_time:.2f} 초 소요")

    # 3. B-spline 그리드 초기화
    step_start_time = time.time()
    print("3단계: B-spline 그리드 초기화")
    grid_physical_spacing = [50.0, 50.0]  # 각 그리드 포인트 간격 설정
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
    registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=0.5, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetInitialTransform(initial_transform, inPlace=True)
    registration_method.SetInterpolator(sitk.sitkLinear)

    # 멀티스케일 접근법 설정
    # registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

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
    cv2.imwrite(output_image_path, out_array)
    step_end_time = time.time()
    print(f"6단계 완료: {step_end_time - step_start_time:.2f} 초 소요")
    print(f"결과 이미지가 {output_image_path}에 저장되었습니다.")

    overlayed_image = overlay_images(fixed_image_cv, out_array)
    cv2.imwrite('./lung3/nonrigid_overlayed_image.jpg', overlayed_image)

    # 정합 후 유사도 계산
    rigid_ssim = ssim(fixed_image_cv, moved_image_cv) *100
    final_ssim = ssim(fixed_image_cv, out_array) *100  
    print(f"정합 전 SSIM: {rigid_ssim:.2f}%")
    print(f"정합 후 SSIM: {final_ssim:.2f}%")

# 사용 예시
fixed_image_path = './lung3/lung3_fixed.jpg'  # 고정 이미지 파일 경로
moved_image_path = './lung3/lung3_aligned.jpg'  # 이동 이미지 파일 경로
output_image_path = './lung3/final_aligned_image.jpg'  # 결과 이미지 저장 경로

b_spline_registration(fixed_image_path, moved_image_path, output_image_path)