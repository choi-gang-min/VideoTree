import json

# JSON 파일 읽기
with open('gpt4mini/cluster_2_4/eval_qa.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 새로운 딕셔너리 생성
new_data = {}

# 각 항목을 순회하며 키 수정
for key, value in data.items():
    # qid 값 가져오기
    qid = value.get('qid', '')
    
    # 새로운 키 생성 (기존키_qid)
    new_key = f"{key}_{qid}"
    
    # 새로운 딕셔너리에 추가
    new_data[new_key] = value

# 수정된 데이터를 새 파일로 저장
with open('eval_qa_modified.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

print(f"처리 완료: {len(data)}개 항목 → {len(new_data)}개 항목")
print("저장 위치: eval_qa_modified.json")