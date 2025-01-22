import os
import glob
import shutil


def copy_files(file_paths, target_directories, new_names):
    # 모든 리스트의 길이 확인
    if len(file_paths) != len(target_directories) or len(file_paths) != len(new_names):
        raise ValueError("file_paths, target_directories, new_names의 길이가 같아야 합니다.")

    for file_path, target_directory, new_name in zip(file_paths, target_directories, new_names):
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)  # 디렉토리가 없으면 생성

        if os.path.exists(file_path):
            target_path = os.path.join(target_directory, new_name)
            shutil.copy2(file_path, target_path)
            print(f"파일 복사 및 이름 변경 완료: {file_path} -> {target_path}")
        else:
            print(f"파일을 찾을 수 없음: {file_path}")

def split_filename_with_path(filepath):
    # 파일 경로에서 파일 이름만 추출
    filename = filepath.split('\\')[-1]  # Windows 경로 구분자 '\\'로 분리

    # '.'로 분리하여 확장자 분리
    name_parts = filename.rsplit('.', 1)
    name_without_extension = name_parts[0]
    extension = '.' + name_parts[1]

    # '_'로 분리하여 나머지 부분 분리
    name_main, number = name_without_extension.rsplit('_', 1)

    # 결과 반환
    return name_main, number, extension

real_image_dataset_folder = "E:\\Resources\\IndoorSceneSynthesis\\InstructScene\\threed_front_bedroom\\text_file_name\\blender_rendered_scene_256\\text_file_number"
target_image_dataset_folder = "E:\\Resources\\IndoorSceneSynthesis\\InstructScene\\threed_front_bedroom\\text_file_name\\"
text_dataset_folder = "C:\\Users\\ttd85\\Downloads\\outputs"
text_file_paths = glob.glob(os.path.join(text_dataset_folder, "*.txt"))

for text_file_path in text_file_paths:
    text_file_name, text_file_number, text_file_extension = split_filename_with_path(text_file_path)
    real_image_file_path = real_image_dataset_folder.replace("text_file_name", text_file_name).replace("text_file_number", text_file_number) + ".png"
    target_image_file_path = target_image_dataset_folder.replace("text_file_name", text_file_name)+ "room_mask.png"

    file_paths = [text_file_path, real_image_file_path, target_image_file_path]
    target_directories = ["outputs\\text_files", "outputs\\real_images", "outputs\\target_images"]
    new_names = [text_file_name + ".txt", text_file_name + ".png", text_file_name + ".png"]
    copy_files(file_paths, target_directories, new_names)