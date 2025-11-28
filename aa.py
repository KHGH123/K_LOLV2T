import json

# 입력 JSON 파일 경로
input_file = r"annotations\youcook2\validation.json"
# 결과 저장할 JSON 파일 경로
output_file = r"annotations\youcook2\captioning_val_para.json"

# 1. JSON 파일 읽기
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. sentences 합치기
result = {}
for key, value in data.items():
    sentences = value.get("sentences", [])
    merged = " ".join(sentences)
    result[key] = merged

# 3. 결과 저장
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"Saved processed JSON to {output_file}")
