import os
from huggingface_hub import login
with open("api_key.txt", "r") as f:
    api_key = f.read().strip()  # 공백 제거
os.environ["HF_TOKEN"] = api_key
login(api_key)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 베이스 모델 경로 혹은 허브 모델 ID
base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
# 학습된 adapter가 저장된 디렉토리 (adapter_config.json, adapter_model.safetensors, README.md 가 위치한 폴더)
adapter_path = "Qwen2.5-Coder-7B-Instruct"

# 토크나이저와 베이스 모델 로드
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)

# adapter 로드: 베이스 모델 위에 학습된 adapter를 적용합니다.
model = PeftModel.from_pretrained(model, adapter_path)

# 테스트를 위한 메시지 구성 (채팅 템플릿 사용)
prompt = """
I want to generate layout in bedroom style. Please generate the layout according to the remaining values I provide:
```
<html>
    <body>
        <rect data-category=dressing_table caption="a wooden dressing table with a drawer and a mirror" transform="translate3d(<FILL_x>, <FILL_y>, <FILL_z>) scale3d(<FILL_x>, <FILL_y>, <FILL_z>) rotateY(<FILL_deg>)"/>
        <rect data-category=nightstand caption="a nightstand with three drawers" transform="translate3d(<FILL_x>, <FILL_y>, <FILL_z>) scale3d(<FILL_x>, <FILL_y>, <FILL_z>) rotateY(<FILL_deg>)"/>
        <rect data-category=stool caption="a beige cushioned stool with gold trim" transform="translate3d(<FILL_x>, <FILL_y>, <FILL_z>) scale3d(<FILL_x>, <FILL_y>, <FILL_z>) rotateY(<FILL_deg>)"/>
        <rect data-category=double_bed caption="a grey double bed with headboard and pillows" transform="translate3d(<FILL_x>, <FILL_y>, <FILL_z>) scale3d(<FILL_x>, <FILL_y>, <FILL_z>) rotateY(<FILL_deg>)"/>
        <rect data-category=cabinet caption="a console table with doors and drawers" transform="translate3d(<FILL_x>, <FILL_y>, <FILL_z>) scale3d(<FILL_x>, <FILL_y>, <FILL_z>) rotateY(<FILL_deg>)"/>
        <rect data-category=pendant_lamp caption="a pendant lamp with metal cups" transform="translate3d(<FILL_x>, <FILL_y>, <FILL_z>) scale3d(<FILL_x>, <FILL_y>, <FILL_z>) rotateY(<FILL_deg>)"/>
        <rect data-category=wardrobe caption="a wardrobe with clothes" transform="translate3d(<FILL_x>, <FILL_y>, <FILL_z>) scale3d(<FILL_x>, <FILL_y>, <FILL_z>) rotateY(<FILL_deg>)"/>
    </body>
</html>
```
"""
messages = [
    {"role": "user", "content": prompt}
]

# 채팅 템플릿을 적용해 프롬프트 생성 (미리 정의된 tokenizer 함수 사용)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 프롬프트 토큰화 후 모델 추론 (GPU 사용 가정)
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150, num_return_sequences=1)

# 생성된 텍스트 디코딩 및 어시스턴트 응답 추출
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# adapter를 통해 학습된 어시스턴트 응답 부분 출력 (모델에 따라 텍스트 형식이 다를 수 있음)
print("Assistant response:", text.split("assistant")[-1])