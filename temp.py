memory_storage = []
prev_ms = [1]  # 1. 처음엔 1이 들어있음


memory_storage.append(prev_ms) # 저장! (우리는 1이 저장됐다고 생각함)
print(memory_storage)
# 2. 다음 스텝에서 prev_ms의 내용을 바꿈 (리스트 재사용)
prev_ms[0] = 99 

print(memory_storage)