import os
from bert_score import score
from huggingface_hub import login

with open("api_key.txt", "r") as f:
    api_key = f.read().strip()  # 공백 제거
os.environ["HF_TOKEN"] = api_key
login(api_key)

# 폴더 경로 설정
folder_path = "./test_outputs"

# 파일 목록 가져오기
file_list = os.listdir(folder_path)

real_texts = [file for file in file_list if "real_text.txt" in file]
target_texts = [file for file in file_list if "target_text.txt" in file]

real_outputs = []
target_outputs = []
for real_text, target_text in zip(real_texts, target_texts):
    with open(os.path.join(folder_path, real_text), "r", encoding='UTF-8') as f:
        real_output = f.read()

    with open(os.path.join(folder_path, target_text), "r", encoding='UTF-8') as f:
        target_output = f.read()

    real_outputs.append(real_output)
    target_outputs.append(target_output)

print(len(real_outputs))
# BERTScore 계산: 한국어 평가를 위해 lang="ko"를 지정하고, 다국어 지원 모델 사용.
P, R, F1 = score(
    target_outputs,
    real_outputs,
    lang="en",
    verbose=True
)

# 계산된 F1 점수를 출력합니다.
print("BERTScore F1:", F1)
print(f"System level F1 score: {F1.mean():.3f}")