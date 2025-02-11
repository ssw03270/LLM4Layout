import os
from bert_score import score
from huggingface_hub import login

with open("api_key.txt", "r") as f:
    api_key = f.read().strip()  # 공백 제거
os.environ["HF_TOKEN"] = api_key
login(api_key)

# 폴더 경로 설정
folder_path = "outputs/no_test_outputs"
asdf_folder_path = "outputs/test_outputs"

# 파일 목록 가져오기
file_list = os.listdir(folder_path)

real_texts = [file for file in file_list if "real_text.txt" in file]
target_texts = [file for file in file_list if "target_text.txt" in file]

real_outputs = []
target_outputs = []

real_output_firsts = []
real_output_seconds = []
real_output_thirds = []
real_output_forths = []

target_output_firsts = []
target_output_seconds = []
target_output_thirds = []
target_output_forths = []

for real_text, target_text in zip(real_texts, target_texts):
    with open(os.path.join(asdf_folder_path, real_text), "r", encoding='UTF-8') as f:
        real_output = f.read()

    with open(os.path.join(folder_path, target_text), "r", encoding='UTF-8') as f:
        target_output = f.read()

    real_outputs.append(real_output)
    target_outputs.append(target_output)

    real_output_split = real_output.split(".\n\n")
    target_output_split = target_output.split(".\n\n")

    real_length = len(real_output_split)
    target_length = len(target_output_split)

    if real_length > 6 or real_length < 4:
        continue
    if target_length > 6 or target_length < 4:
        continue

    if real_length == 4:
        real_output_firsts.append(real_output_split[0])
        real_output_seconds.append(real_output_split[1])
        real_output_thirds.append(real_output_split[2])
        real_output_forths.append(real_output_split[3])
    if real_length == 5:
        real_output_firsts.append(real_output_split[1])
        real_output_seconds.append(real_output_split[2])
        real_output_thirds.append(real_output_split[3])
        real_output_forths.append(real_output_split[4])
    if real_length == 6:
        real_output_firsts.append(real_output_split[1])
        real_output_seconds.append(real_output_split[2])
        real_output_thirds.append(real_output_split[3])
        real_output_forths.append(real_output_split[4])


    if target_length == 4:
        target_output_firsts.append(target_output_split[0])
        target_output_seconds.append(target_output_split[1])
        target_output_thirds.append(target_output_split[2])
        target_output_forths.append(target_output_split[3])
    if target_length == 5:
        target_output_firsts.append(target_output_split[1])
        target_output_seconds.append(target_output_split[2])
        target_output_thirds.append(target_output_split[3])
        target_output_forths.append(target_output_split[4])
    if target_length == 6:
        target_output_firsts.append(target_output_split[1])
        target_output_seconds.append(target_output_split[2])
        target_output_thirds.append(target_output_split[3])
        target_output_forths.append(target_output_split[4])

P_total, R_total, F1_total = score(real_outputs, target_outputs, lang="en", verbose=True)
P_first, R_first, F1_first = score(real_output_firsts, target_output_firsts, lang="en", verbose=True)
P_second, R_second, F1_second = score(real_output_seconds, target_output_seconds, lang="en", verbose=True)
P_third, R_third, F1_third = score(real_output_thirds, target_output_thirds, lang="en", verbose=True)
P_fourth, R_fourth, F1_fourth = score(real_output_forths, target_output_forths, lang="en", verbose=True)

# 각 데이터셋의 길이 계산
len_total = len(real_outputs)
len_first = len(real_output_firsts)
len_second = len(real_output_seconds)
len_third = len(real_output_thirds)
len_fourth = len(real_output_forths)

# 결과 출력 (F1은 소수점 4자리까지)
print("전체 데이터 F1: {:.4f}, 길이: {}".format(F1_total.mean(), len_total))
print("첫 번째 데이터 F1: {:.4f}, 길이: {}".format(F1_first.mean(), len_first))
print("두 번째 데이터 F1: {:.4f}, 길이: {}".format(F1_second.mean(), len_second))
print("세 번째 데이터 F1: {:.4f}, 길이: {}".format(F1_third.mean(), len_third))
print("네 번째 데이터 F1: {:.4f}, 길이: {}".format(F1_fourth.mean(), len_fourth))