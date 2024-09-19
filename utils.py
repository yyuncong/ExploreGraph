import shutil
import os
import json
from tqdm import tqdm
import random

def merge_datasets(merge_from, merge_to):
    questions_from = os.listdir(merge_from)
    print(merge_from)
    print(len(questions_from))
    if 'log.log' in questions_from:
        print('yes')
    questions_to = os.listdir(merge_to)
    print(merge_to)
    print(len(questions_to))
    print(len(set(questions_from)&set(questions_to)))
    
    for qf in questions_from:
        frompath = os.path.join(merge_from, qf)
        if not os.path.isdir(frompath):
            continue
        topath = os.path.join(merge_to,qf)
        if os.path.exists(topath):
            print(f"Removing conflicting directory: {topath}")
            #shutil.rmtree(topath)

def checkclass_set(obj_bbox_dir, map_path):
    obj_bbox_files = os.listdir(obj_bbox_dir)
    obj_bbox_files = [f for f in obj_bbox_files if f.endswith('.json')]
    obj_class_set = set()
    for obj_bbox_file in obj_bbox_files:
        with open(os.path.join(obj_bbox_dir, obj_bbox_file), 'r') as f:
            obj_bbox = json.load(f)
        for obj in obj_bbox:
            obj_class = obj['class_name']
            obj_class_set.add(obj_class)
    with open(map_path, 'r') as f:
        class_map = json.load(f)
    class_set = set(class_map.keys())
    print(len(obj_class_set))
    print(len(class_set))
    print(len(obj_class_set&class_set))

def check_sene_set(src):
    scene_set = set()
    valid_file_num = 0
    for subdir in tqdm(os.listdir(src)):
        try:
            with open(os.path.join(src,subdir,'metadata.json'),"r") as f:
                metadata = json.load(f)
        except:
            continue
        scene = metadata["scene"]
        scene_set.add(scene)
        valid_file_num += 1
    print(sorted(list(scene_set)))
    print(f"valid file num: {valid_file_num}")
        
def collect_one_example(src):
    
    #sample_id = random.choice(os.listdir(src))
    sample_id = sorted(os.listdir(src))[0]
    print(sample_id)
    # collect metadata
    sample_path = os.path.join(src,sample_id)
    with open(os.path.join(sample_path,'metadata.json'),"r") as f:
        metadata = json.load(f)
    print(sorted(os.listdir(sample_path)))
    step_path = os.path.join(sample_path, '0000.json')
    with open(step_path, "r") as f:
        step = json.load(f)
    print(step)
        
    
if __name__ == "__main__":
    '''
    exploration_path = "/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/3d/explore-eqa-test/"
    merge_datasets(
        os.path.join(exploration_path,'exploration_data_2.5_best'),
        os.path.join(exploration_path,'exploration_data_2.5_best_fixed')
    )
    '''
    #obj_bbox_dir = "/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d/hm3d_obj_bbox_all"
    #map_path = "bbox_mapping/matterport_category_map.json"
    #checkclass_set(obj_bbox_dir, map_path)
    #exploration_path = "/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/3d/explore-eqa-test/"
    #check_sene_set(os.path.join(exploration_path,'exploration_data_goatbench'))
    exploration_path = "/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/3d/explore-eqa-test/"
    exploration_data = os.path.join(exploration_path, "exploration_data")
    collect_one_example(exploration_data)
    
    