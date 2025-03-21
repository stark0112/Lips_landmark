import torch

# 모델 파일 로드
full_model_path = "blazeface_full_best.pth"
full_model_state = torch.load(full_model_path, map_location="cpu")

# 확인할 레이어 목록
layers_to_check = [
    "bbox_head.weight", "bbox_head.bias",
    "conf_head.weight", "conf_head.bias",
    "landmark_subnet.0.weight", "landmark_subnet.0.bias",
    "landmark_subnet.2.weight", "landmark_subnet.2.bias"
]


# 각 레이어의 값 출력 (샘플 5개만)
print(f"🔍 {full_model_path} 내용:")
for layer in layers_to_check:
    if layer in full_model_state:
        layer_data = full_model_state[layer].numpy()
        sample_values = layer_data.flatten()[:20]  # 🔥 샘플 5개만 가져오기
        print(f"{layer} 값 (샘플 5개):\n", sample_values, "\n")

    else:
        print(f"❌ {layer} 키가 존재하지 않습니다.\n")

import torch

# 모델 파일 로드
celeba_model_path = "blazeface_celeba.pth"
celeba_model_state = torch.load(celeba_model_path, map_location="cpu")

# 확인할 레이어 목록
layers_to_check = [
    "bbox_head.weight", "bbox_head.bias",
    "conf_head.weight", "conf_head.bias"
]

# 각 레이어의 값 출력 (샘플 5개만)
print(f"🔍 {celeba_model_path} 내용:")
for layer in layers_to_check:
    if layer in celeba_model_state:
        layer_data = celeba_model_state[layer].numpy()
        sample_values = layer_data.flatten()[:20]  # 🔥 샘플 5개만 가져오기
        print(f"{layer} 값 (샘플 5개):\n", sample_values, "\n")
    else:
        print(f"❌ {layer} 키가 존재하지 않습니다.\n")

'''
import torch

# 저장된 모델 파일 로드
model_path = "blazeface_full_best.pth"
state_dict = torch.load(model_path, map_location="cpu")

# ✅ 모델의 키(레이어 이름) 확인
print("📌 모델의 state_dict 키 목록:")
for key in state_dict.keys():
    print(key)

# ✅ 특정 레이어의 가중치 확인 (예: bbox_head의 첫 번째 Conv 레이어)
if "bbox_head.weight" in state_dict:
    print("\n📌 bbox_head.weight 가중치 샘플:")
    print(state_dict["bbox_head.weight"])
'''

