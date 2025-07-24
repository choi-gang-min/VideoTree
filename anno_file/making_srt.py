import json

with open("/data/jongseo/project/vlm/VideoTree/anno_file/lavila_subset.json", 'r', encoding='utf-8') as f:
    video_captions = json.load(f)
# 예시 데이터 구조: 영상 ID(key)와 캡션 리스트(value)

def create_srt_captions(captions, seconds_per_caption=1):
    """
    캡션 리스트를 받아 1초 단위 SRT 자막 문자열 생성 함수
    """
    srt_strings = []
    
    def sec_to_srt_time(sec):
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        ms = 0
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    
    for i, caption in enumerate(captions):
        index = i + 1
        start_sec = i * seconds_per_caption
        end_sec = (i + 1) * seconds_per_caption
        
        start_time = sec_to_srt_time(start_sec)
        end_time = sec_to_srt_time(end_sec)
        
        srt_block = f"{index}\n{start_time} --> {end_time}\n{caption}"
        srt_strings.append(srt_block)
    
    return "\n\n".join(srt_strings)


# 실제 사용 예시
video_id = "0f9d1d44-8a15-4135-a77e-d64064afc1e4"
captions = video_captions[video_id]

srt_content = create_srt_captions(captions)
import os
# srt_content를 파일로 저장하려면 아래 코드 사용
with open(os.path.join('anno_file/srt',f"{video_id}.srt"), "w", encoding="utf-8") as f:
    f.write(srt_content)

print(f"SRT 파일이 {video_id}.srt 이름으로 저장되었습니다.")
