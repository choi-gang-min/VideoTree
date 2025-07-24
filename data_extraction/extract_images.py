import cv2
from pathlib import Path
from tqdm import tqdm
import json
import pandas as pd


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


def extract_es():
    input_base_path = Path('/data/dataset/video_tree/data/intentqa/videos')
    input_csv_path = Path('/data/dataset/video_tree/data/intentqa/test_id.csv')
    output_base_path = Path('/data/dataset/intentqa_videotree/frames/')
    fps = 1
    
    # CSV 파일에서 video_id 목록 읽기
    df = pd.read_csv(input_csv_path)
    test_video_ids = df['video_id'].astype(str).tolist()
    test_video_ids_set = set(test_video_ids)
    
    # 비디오 파일 목록
    all_video_files = list(input_base_path.glob('*.mp4'))
    video_files = [f for f in all_video_files if f.stem in test_video_ids_set]
    
    print(f"총 {len(video_files)}개의 test 비디오를 처리합니다.")
    
    # 에러 발생한 파일들을 추적
    error_files = []
    success_count = 0
    
    pbar = tqdm(total=len(video_files))
    for video_fp in video_files:
        try:
            output_path = output_base_path / video_fp.stem
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 비디오 열기
            vidcap = cv2.VideoCapture(str(video_fp))
            
            # 비디오가 제대로 열렸는지 확인
            if not vidcap.isOpened():
                error_files.append((video_fp.name, "Failed to open video"))
                pbar.update(1)
                continue
            
            fps_ori = vidcap.get(cv2.CAP_PROP_FPS)
            if fps_ori == 0:
                error_files.append((video_fp.name, "FPS is 0"))
                vidcap.release()
                pbar.update(1)
                continue
                
            frame_interval = int(fps_ori / fps)
            
            count = 0
            saved_frames = 0
            
            while True:
                success, image = vidcap.read()
                if not success:
                    break
                    
                if count % frame_interval == 0:
                    try:
                        # 프레임 번호 대신 저장된 프레임 수를 사용
                        output_filename = f'{output_path}/{saved_frames:06d}.jpg'
                        cv2.imwrite(output_filename, image)
                        saved_frames += 1
                    except Exception as e:
                        error_files.append((video_fp.name, f"Frame write error: {str(e)}"))
                        break
                        
                count += 1
            
            vidcap.release()
            success_count += 1
            
        except Exception as e:
            error_files.append((video_fp.name, str(e)))
            
        pbar.update(1)
    
    pbar.close()
    
    # 결과 출력
    print(f"\n처리 완료: 성공 {success_count}개, 실패 {len(error_files)}개")
    
    if error_files:
        print("\n에러 발생한 파일들:")
        for filename, error in error_files[:10]:  # 처음 10개만 출력
            print(f"  - {filename}: {error}")
        
        # 에러 로그 저장
        error_log_path = output_base_path / 'error_log.txt'
        with open(error_log_path, 'w') as f:
            for filename, error in error_files:
                f.write(f"{filename}: {error}\n")
        print(f"\n전체 에러 로그는 {error_log_path}에 저장되었습니다.")


if __name__ == '__main__':
    extract_es()