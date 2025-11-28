import json
import csv

json_files = [r"annotations\youcook2\training.json", r"annotations\youcook2\validation.json"]
output_file = "output.csv"

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter='\t')  # 탭 구분
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            for key, value in data.items():
                duration = value["duration"]
                frames = round(duration * 30)
                # 각 값이 열로 나뉘도록 리스트로 전달
                writer.writerow([key, duration, frames])

print(f"CSV 파일 생성 완료: {output_file}")
