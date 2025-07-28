import os
from pathlib import Path
from util import *
from eval import *
from dataset import get_dataset
from prompts import PromptFactory
from model import get_model
from tqdm import tqdm
from pprint import pprint
from kmeans_pytorch import kmeans
import torch

def load_frame_features(name_ids, save_folder):
    """
    Load frame features from a .pt file.

    Args:
    - filename (str): Name of the .pt file to load

    Returns:
    - img_feats (torch.Tensor): Loaded image features
    """
    filename = f"{name_ids}.pt"  # Construct filename with name_ids
    filepath = os.path.join(save_folder, filename)
    img_feats = torch.load(filepath)
    return img_feats

def find_closest_points_per_cluster(x, cluster_ids, cluster_centers):
    # Dictionary to store the indices of the closest points for each cluster
    closest_points_idx_per_cluster = {cluster_id: [] for cluster_id in range(len(cluster_centers))}
    
    # Iterate over each cluster
    for cluster_id in range(len(cluster_centers)):
        # Filter points belonging to the current cluster
        indices_in_cluster = torch.where(cluster_ids == cluster_id)[0]
        points_in_cluster = x[indices_in_cluster]
        
        # Calculate distances from points in the cluster to the cluster center
        distances = torch.norm(points_in_cluster - cluster_centers[cluster_id], dim=1)

        if distances.numel() > 0:    
            
            # Find the index (within the cluster) of the point closest to the cluster center
            closest_idx_in_cluster = torch.argmin(distances).item()
            
            # Map back to the original index in x
            closest_global_idx = indices_in_cluster[closest_idx_in_cluster].item()
            
            # Store the global index
            closest_points_idx_per_cluster[cluster_id].append(closest_global_idx)

    return closest_points_idx_per_cluster

def launch():
    args = parse_args()
    pprint(args)

        # output
    makedir(args.output_base_path)
    output_path = os.path.join(args.output_base_path, args.output_filename)
    output_width_res_path = os.path.join(args.output_base_path,"width_res.json")
    frame_feat_path = args.frame_feat_path

    # resume
    processed = {}
    if not args.start_from_scratch and os.path.exists(output_path):
        processed = load_json(output_path)
        if 'data' in processed:
            processed = processed['data']

    # get input
    quids_to_exclude = set(list(processed.keys()))
    dataset = get_dataset(args, quids_to_exclude=quids_to_exclude, num_examples_to_run=args.num_examples_to_run)

    # configure prompt
    prompter = PromptFactory().get(args.prompt_type)

    # get model
    model = get_model(args)
    model.set_post_process_fn(prompter.post_process_fn)
    
    # save width expansion results
    all_width_res = []

    # answer
    pbar = tqdm(total=len(dataset))
    for i, item in enumerate(dataset):
        ukey_1 = item['quid'] if 'quid' in item else item['uid']
        
        # 어디 비디오에서 삑나나 보자
        print(f"\nProcessing video: {ukey_1}")
        print("QuestionID : ",item['qid'])
        print("Question : ", item['question'])
        

        #init the cluster parameters
        tree_node = [0]
        max_cluster_num = args.max_cluster_num
        cluster_num = args.init_cluster_num
        iter_threshold = args.iter_threshold
        adaptive_rate = args.default_adpative_rate


        clip_length = int(1/args.fps) if args.fps < 1 else 1/args.fps
        few_shot_examples = build_fewshot_examples(args.fewshot_example_path, args.data_path)

        # load frame features
        frame_feats = load_frame_features(ukey_1, frame_feat_path)
        
        frame_length = frame_feats.shape[0]
        print(f"Length of Frame : {frame_length} ")
        ### adaptive width expansion
        while(True):
            # width expansion
            cluster_ids_x, cluster_centers = kmeans(X=frame_feats, num_clusters=cluster_num, distance='cosine', device=torch.device('cuda:0'))
            # send cluster_ids_x to GPU 
            cluster_ids_x = cluster_ids_x.to('cuda')
            cluster_centers = cluster_centers.to('cuda')
            closest_points_idx_per_cluster = find_closest_points_per_cluster(frame_feats, cluster_ids_x, cluster_centers)
            if closest_points_idx_per_cluster is None:
                # print("closest_points_idx_per_cluster is None")
                continue
            tree_node = sorted([value for sublist in closest_points_idx_per_cluster.values() for value in sublist])

            cluster_ids_x = cluster_ids_x.tolist()
            # relevance scoring

            #-------------      
            # IntentQA narration 전처리
            if args.dataset == 'intentqa':
                # narration을 tree_node에 해당하는 프레임만 남기기
                lines = item['narration'].strip().split('\n')
                narration_dict = {}
                for line in lines:
                    if ': ' in line:
                        idx, caption = line.split(': ', 1)
                        narration_dict[int(idx)] = caption
                
                # tree_node에 해당하는 프레임만 선택
                selected_narrations = []
                for idx in tree_node:
                    if idx in narration_dict:
                        selected_narrations.append(f"{idx}: {narration_dict[idx]}")
                
                # item 복사 후 narration 교체
                item_copy = item.copy()
                item_copy['narration'] = '\n'.join(selected_narrations)
            else:
                item_copy = item
            #------------------


            model.set_post_process_fn(prompter.post_process_fn)
            # prompt = prompter.fill(**item, fps=args.fps, clip_length=clip_length, num_words=args.num_words_in_sum, examplars=few_shot_examples, loc_pred = tree_node)

            # 수정 후 (IntentQA일 때만)
            if args.dataset == 'intentqa':
                prompt = prompter.fill(**item_copy, fps=args.fps, clip_length=clip_length, 
                                    num_words=args.num_words_in_sum, examplars=few_shot_examples)
                # loc_pred를 빼기!
            else:
                prompt = prompter.fill(**item_copy, fps=args.fps, clip_length=clip_length, 
                                    num_words=args.num_words_in_sum, examplars=few_shot_examples, 
                                    loc_pred=tree_node)


            # pred, info = model.forward(prompter.head, prompt)
            max_attempts = 10

            fallback_head = (
                "You are presented with a textual description of a video, it consists of N frame captions sparsely sampled from the video "
                "(#C means the first person view, and #O indicates another). "
                "The ultimate goal is to answer a question related to this video, choosing the correct option out of five possible answers. "
                "Please provide the answer with a single-letter (A, B, C, D, E). "
                "It is crucial that you imagine the visual scene as vividly as possible to enhance the accuracy of your response. "
                "After selecting your answer, rate your confidence level in this choice on a scale from 1 to 100, "
                "where 1 indicates low confidence and 100 signifies high confidence. "
                "Please provide a concise one-sentence explanation for your chosen answer. "
                "If you are uncertain about the correct option, select the one that seems closest to being correct. "
                f"\n\nThere are exactly {len(tree_node)} frame captions above. "
                f"Please return exactly {len(tree_node)} relevance scores, one for each frame, "
                "in the format of a list like [1, 2, 3, ..., 1]. "
                "Each number should correspond to the relevance of each frame caption in the same order. "
                "The score is between 1 (low relevance), 2 (medium), and 3 (high relevance). "
                "Please return your final answer in the format below:\n\n"
                "prediction:\n explanation:\n confidence:\n frame relevance:\n"
            )


            # for attempt in range(max_attempts):
            #     pred, info = model.forward(prompter.head, prompt)

            #     frame_relevance = pred
            #     if len(frame_relevance) == len(tree_node):
            #         print("Success -> Attempt:", attempt + 1)
            #         break  # 정상 → 루프 탈출
            #     else:
            #         print(f"[Warning] Relevance score length doesn't match to cluster Number  (Trying {attempt + 1}/{max_attempts})")
            #         print(f"→ tree_node: {len(tree_node)}, pred: {len(frame_relevance)}")
            #         print("Caption Prompt:", prompt[:300])
            #         print("Raw response:", info['response'])

            #         if attempt < max_attempts - 1:
            #             print("→ Retrying...")
            #         else:
            #             raise ValueError(f"Retried {max_attempts} time, but Tree_node and scoring doesn't match. Jump to next video.")

            for attempt in range(max_attempts):
                # 5번째 시도부터는 fallback_head 사용
                if attempt >= 5:
                    current_head = fallback_head
                else:
                    current_head = prompter.head

                pred, info = model.forward(current_head, prompt)

                frame_relevance = pred
                if len(frame_relevance) == len(tree_node):
                    print("✅ Success -> Attempt:", attempt + 1)
                    break
                else:
                    print(f"[⚠️ Warning] Relevance score length mismatch (Attempt {attempt + 1}/{max_attempts})")
                    print(f"→ len(tree_node): {len(tree_node)}, len(pred): {len(frame_relevance)}")
                    print("Caption Prompt:", prompt[:300])
                    print("Raw response:", info['response'][:300])

                    if attempt < max_attempts - 1:
                        print("→ Retrying...")
                    else:
                        raise ValueError(f"❌ Retried {max_attempts} times, but tree_node and relevance scoring still mismatch. Skipping this video.")





            ukey_name = 'quid' if 'quid' in item else 'uid'

            # the output is the predicted frame relevance
            frame_relevance = pred
            high_relevance_frame_num = frame_relevance.count(3)
            
            
            
            # if high_relevance_frame_num < iter_threshold:
            #     if cluster_num < max_cluster_num:
            #         cluster_num = cluster_num * adaptive_rate
            #     else:
            #         break
            # else:
            #     break

            if high_relevance_frame_num < iter_threshold:
                next_cluster_num = int(cluster_num * adaptive_rate)
                
                # frame_length와 max_cluster_num 중 작은 값을 상한선으로
                upper_limit = min(frame_length, max_cluster_num)
                
                if next_cluster_num <= upper_limit:
                    cluster_num = next_cluster_num
                elif cluster_num < upper_limit:
                    # 마지막으로 상한선 값으로 한 번 더 시도
                    cluster_num = upper_limit
                else:
                    # 이미 상한선에 도달했으면 종료
                    break
            else:
                break
            
            
        all_width_res.append({"name": ukey_1, "tree_node": tree_node, "cluster_ids_x": cluster_ids_x})


        ukey = item[ukey_name]

        
        # # ========= 여기에 추가 =========
        # # 고유한 키 생성
        # if args.dataset == 'intentqa':
        #     video_id = item['uid']  # 비디오 ID
        #     qid = item['qid']      # 질문 ID
        #     ukey = f"{video_id}_{qid}"  # "11_2", "12_2" 형태
        # else:
        #     ukey = item[ukey_name]  # 기존대로
        # ===============================
        if args.dataset == 'intentqa':
            video_id = item['uid']
            qid = item['qid']
            ukey = f"{video_id}_{qid}"  # 예: "video_01_3"
        else:
            ukey_name = 'quid' if 'quid' in item else 'uid'
            ukey = item[ukey_name]

        
        processed[ukey] = item

        processed[ukey]['prompt'] = prompt
        processed[ukey]['prompt_template'] = prompter.get_template_str()
        processed[ukey]['response'] = info['response']
        processed[ukey]['pred'] = pred
        if args.save_info:
            processed[ukey]['info'] = {k: v for k, v in info.items() if k != 'response'}
        if i % args.save_every == 0:
            save_json(processed, output_path)
        pbar.update(1)
        print(f"Finished Processing video: {ukey_1}, cluster num: {cluster_num}")

    save_json(all_width_res, output_width_res_path)
    

    # incorporate with backup prediction
    if len(args.backup_pred_path) > 0:
        backup = load_json(args.backup_pred_path)
        if 'data' in backup:
            backup = backup['data']
        for uid in processed:
            if processed[uid]['pred'] == -1:
                # processed[uid]['pred'] = backup[uid]['pred']
                processed[uid]['pred'] = backup[uid]['pred']


    # if eval
    if not args.disable_eval:
        if args.task == 'qa':
            if args.dataset == 'egoschema':
                processed = eval_qa_egoschema(processed)
            elif args.dataset in ['nextqa', 'intentqa', 'nextgqa']:
                processed = eval_qa_nextqa(args.anno_path, processed)
        elif args.task == 'gqa':
            if args.dataset == 'nextgqa':
                pred_qa_path = args.nextgqa_pred_qa_path if len(args.nextgqa_pred_qa_path) > 0 else None
                processed = eval_gqa(args.nextgqa_gt_ground_path, processed, pred_qa_path=pred_qa_path)
        elif args.task == 'sum':
            processed, sum_data = eval_sum(processed)
            save_json(sum_data, f'{Path(output_path).parent / Path(output_path).stem}_data.json')

    save_json(processed, output_path)


if __name__ == '__main__':
    launch()
    