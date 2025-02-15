from torch.utils.data.distributed import DistributedSampler
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from easydict import EasyDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset, Subset
from itertools import chain
import random
import numpy as np
import math

SCENE_TOKEN = "<scene>"
# FRONTIER_TOKEN = "<frontier>"
SELECT_TOKEN = "<select>"
SCENE_TOKEN = "<scene>"
VISUAL_TOKEN = "<visual>"
TACTILE_TOKEN = "<temperature>"
SOUND_TOKEN = "<sound>"
# TEMP_TOKEN = "<temperature>"
GET_VISUAL_TOKEN = "<observe>"
GET_TACTILE_TOKEN = "<touch>"
GET_SOUND_TOKEN = "<tap>"
SELECT_TOKEN = "<select>"


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe


def discretize_coordinates(coords, num_bins=128, coord_range=(-10, 10)):
    # Ensure coords is a torch tensor
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.float32)

    # Extract min and max values from the coord_range
    min_val, max_val = coord_range

    # Normalize coordinates to range [0, 1]
    normalized_coords = (coords - min_val) / (max_val - min_val)

    # Scale normalized coordinates to range [0, num_bins - 1]
    scaled_coords = normalized_coords * (num_bins - 1)

    # Round to get discrete bin indices and clamp to ensure within range
    discretized_coords = torch.round(scaled_coords).long()
    discretized_coords = torch.clamp(discretized_coords, 0, num_bins - 1)

    return discretized_coords


def sum_positional_encodings(x, pos, pe, num_bins=128, coord_range=(-10, 10)):
    """
    x: (num_points, d_model)
    pos: (num_points, 2)
    pe: (d_model, num_bins, num_bins)
    """
    # Discretize the coordinates
    discretized_coords = discretize_coordinates(
        pos, num_bins=num_bins, coord_range=coord_range
    ).unsqueeze(0)
    # Get the positional encodings for the coordinates
    x_pe = (
        pe[:, discretized_coords[:, :, 0], discretized_coords[:, :, 2]]
        .permute(1, 2, 0)
        .squeeze(0)
    )
    # Sum the positional encodings along the num_points dimension
    x += x_pe
    return x


def pad_zero(x, length):
    if len(x) < length:
        x = "".join(["0" for _ in range(length - len(x))]) + x
    return x


def show_sample(sample):
    for k, v in sample.items():
        print(k, v)
        if not isinstance(v, list):
            print(v.shape)


def prepare_egocentric_view(egocentric_path, visual_feature_size, patch_size):
    #text = "Followings are the egocentric views(in left, right and forward directions):\n "
    text = "Followings are the egocentric views:\n "
    num_tokens = (visual_feature_size // patch_size) ** 2
    egocentric_features = []
    for i, view in egocentric_path.items():
        egocentric_feature = torch.load(view, map_location="cpu")
        egocentric_feature = merge_patches(
            egocentric_feature.view(visual_feature_size, visual_feature_size, -1),
            patch_size,
        )
        egocentric_features.append(egocentric_feature)
        for _ in range(num_tokens):
            text += f"<scene>"
    egocentric_features = torch.cat(egocentric_features, dim=0)
    text += " /\n"
    return text, egocentric_features


def prepare_action_memory(memory_path):
    text = f"Here is your selection in the previous step:\n "
    if memory_path is None:
        text += f"No selection in the previous step. "
        memory_feature = None
    else:
        memory_feature = torch.load(memory_path, map_location="cpu")
        text += f"<scene> "
    text += "/\n"
    return text, memory_feature


def prepare_frontier(feature_path, frontier_info, visual_feature_size, patch_size):
    # print("frontier after shuffle", [info['rgb_id'] for info in frontier_info])
    try:
        text = f"Below are all the frontiers that we can explore.\n"
        num_tokens = (visual_feature_size // patch_size) ** 2
        if len(frontier_info) > 0:
            frontier_features = []
            for i, info in enumerate(frontier_info):
                text += f"frontier {i} "
                #text += f"{i} "
                frontier_feature = torch.load(
                    feature_path[info["rgb_id"]], map_location="cpu"
                )
                frontier_feature = merge_patches(
                    frontier_feature.view(visual_feature_size, visual_feature_size, -1),
                    patch_size,
                )
                frontier_features.append(frontier_feature)
                for _ in range(num_tokens):
                    text += f"<scene>"
                text += " / "
            frontier_features = torch.cat(frontier_features, dim=0)
        else:
            text += f"No frontier available "
            frontier_features = None
        text += "\n"
        return text, frontier_features
    except:
        return None, None


def prepare_prefiltering_input(question, tokenizer, classes, ranking, max_length, topk):
    filter_text = f"Question: {question}\n"
    filter_text += "These are the objects available in current scene graph\n"
    for class_name in classes:
        filter_text += f"{class_name} \n"
    if len(classes) == 0:
        filter_text += "No object available \n"
    # only require selection when there are more than k objects
    # filter_text += f"Select the top {len(ranking)} important objects\n"
    filter_text += f"Rank at most top {topk} of them from high to low based on their importance on answering the question\n"
    # Jiachen TODO 5: format the filtering answer
    answer = "\n".join(ranking[:topk]) if len(classes) > 0 else "No object available"
    filter_text += "Answer: "
    filter_text += answer + tokenizer.eos_token
    # print("filtering prompt", len(filter_text))
    # print(filter_text)
    # Jiachen TODO 7: output filter_input_ids/filter_attention_mask/filter_length for the filtering question
    filter_text = tokenizer(
        filter_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    filter_input_ids = filter_text["input_ids"]
    filter_length = torch.nonzero(filter_input_ids).shape[0]
    filter_attention_mask = filter_text["attention_mask"]
    return filter_input_ids, filter_length, filter_attention_mask


# Merge the 6*6 patches to 2*2 patches
# 1 scene is associated with 4 patches
def merge_patches(patches, patch_size):
    num_patches, num_patches, patch_dim = patches.shape
    #print(patches.shape)
    new_num_patches = num_patches // patch_size
    assert num_patches % patch_size == 0
    patches = patches.view(
        new_num_patches,
        patch_size,
        new_num_patches,
        patch_size,
        patch_dim,
    )
    patches = (
        patches.permute(0, 2, 1, 3, 4)
        .reshape(new_num_patches, new_num_patches, patch_size**2, patch_dim)
        .mean(-2)
    )
    patches = patches.view(new_num_patches * new_num_patches, patch_dim)
    return patches


def format_questions(
    metadata, 
    augment_prompt = False,
    augment_question = False,
    augmented_questions = None,
    image_prompt_visual_feature_size = 12,
    image_prompt_patch_size = 1
    ):
    
    text,question_feature = '', None
    if augment_prompt:
        text += "Task: You are an agent in an indoor scene tasked with answering quesions by observing the surroundings and exploring the environment. To answer the question, you are required to choose either a snapshot or a frontier based on the egocentric views of your surroundings.\n\
        Definitions:\n\
        Snapshot: A focused observation of several objects. Choosing a snapshot means that you are selecting the observed objects in the snapshot as the target objects to help answer the question.\n\
        Frontier: An unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction.\n"
    if "task_type" not in metadata.keys() or metadata["task_type"] == "description":
        question = metadata["question"]
        if augment_question and question in augmented_questions.keys():
            question = np.random.choice(augmented_questions[question])
            #print(f"raw question {raw_question} phrased question {phrased_question}")
        text += f"Question: {question}\n"
    elif metadata["task_type"] == "image":
        #text += "Question: Could you find the object presented in the following image\n"
        text += "Question: Could you find the object captured in the following image?\n"
        try:
            image_feature = torch.load(metadata["image_path"].replace(".png","_full.pt"), map_location="cpu")
        except:
            return None,None
        num_tokens = (image_prompt_visual_feature_size // image_prompt_patch_size) ** 2
        question_feature = merge_patches(
            image_feature.view(
                image_prompt_visual_feature_size,
                image_prompt_visual_feature_size,
                -1,
            ),
            image_prompt_patch_size,
        )
        for _ in range(num_tokens):
            text += "<scene>"
        text += "?/\n"
    elif metadata["task_type"] == "object":
        text += f"Question: Where is the {metadata['target_obj_class']}?\n"
    return text, question_feature
        
        

class ExploreDataset(Dataset):
    def __init__(
        self,
        scene_path,
        exploration_path,
        tokenizer,
        max_length,
        scene_token=SCENE_TOKEN,
        select_token=SELECT_TOKEN,
        egocentric_views=False,
        action_memory=False,
        prefiltering=False,
        random_permute=False,
        add_positional_encodings=False,
        mix_gt=False,
        augment_question=False,
        target_use_gt=False,
        top_k_categories=5,
        num_egocentric_views=5,
        patch_size=3,
        visual_feature_size=6,
        image_prompt_visual_feature_size = 24,
        image_prompt_patch_size = 2,
        gt_rate=0,
        split="train",
    ):
        self.scene_dir = os.path.join(scene_path, "scene_feature_dict_merged_snapshots")
        #self.ranking_path = os.path.join(scene_path, "selected_candidates.json")
        self.ranking_path = "prefiltering/selected_candidates_goatbench.json"
        #self.obj_bbox_dir = "/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d/hm3d_obj_bbox_merged"
        self.obj_bbox_dir ="/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d/hm3d_obj_bbox_all"
        self.explore_dir = os.path.join(exploration_path, "exploration_data_goatbench")
        self.category_map_path = "bbox_mapping/mpcat40_full_map.json"
        self.augmented_questions_path = "question_augment/augmented_generated_questions.json"
        with open(self.category_map_path, "r") as f:
            self.category_map = json.load(f)
        self.tokenizer = tokenizer
        self.scene_token = scene_token
        self.scene_token_id = self.tokenizer(self.scene_token).input_ids[-1]
        self.egocentric_views = egocentric_views
        self.action_memory = action_memory
        self.prefiltering = prefiltering
        self.random_permute = random_permute
        self.augment_question = augment_question
        self.num_egocentric_views = num_egocentric_views
        self.top_k_categories = top_k_categories
        self.mix_gt = mix_gt
        self.gt_rate = gt_rate
        self.target_use_gt = target_use_gt
        

        self.max_length = max_length
        self.split = split
        self.data = self.load_data()

        train_index, test_index = self.split_index()
        self.indices = train_index if split == "train" else test_index
        self.obj_not_found_indices = set({})
        self.too_many_objects_indices = set({})
        self.too_long_prompts_indices = set({})
        self.img_prompt_not_found_indices = set({})
        self.answer_obj_filtered_indices = set({})
        self.bounds = (-7, 7)
        self.num_bins = 128
        self.positional_encoding = positionalencoding2d(
            1024, self.num_bins, self.num_bins
        )
        self.add_positional_encodings = add_positional_encodings

        self.patch_size = patch_size
        assert visual_feature_size % self.patch_size == 0
        # each object is represented by 4 visual tokens(by default)
        self.num_visual_tokens = (visual_feature_size // self.patch_size) ** 2
        self.visual_feature_size = visual_feature_size
        self.image_prompt_visual_feature_size = image_prompt_visual_feature_size
        self.image_prompt_patch_size = image_prompt_patch_size
        

    def load_step(self, step_path):
        try:
            with open(step_path, "r", encoding='utf-8') as f:
                stepdata = json.load(f)
        except Exception as e:
            print("step file loading error")
            print(f"Error loading data at location {step_path}: {e}")
            index = np.random.choice(self.indices)
            print(f"new index loaded {index}")
            new_step_path, new_epi_id = self.data[index]
            print(f"new step path attempted {new_step_path}")
            return self.__getitem__(index)
        epi_path = "/".join(step_path.split("/")[:-1])
        step_file_name = step_path.split("/")[-1]
        step = int(step_file_name.split(".")[0])

        # add paths for frontiers
        stepdata["frontier_features"] = {}
        stepdata["position"] = np.array(stepdata["agent_state"]["init_pts"])[None,]
        # TODO: Need to fix this to make it robust
        try:
            stepdata["frontier_positions"] = (
                np.array([f["coordinate"] for f in stepdata["frontiers"]])
                - stepdata["position"]
            )
        except:
            stepdata["frontier_positions"] = np.array(
                [stepdata["position"] for f in stepdata["frontiers"]]
            )
        frontier_folder = os.path.join(epi_path, "frontier_rgb")
        for frontier in stepdata["frontiers"]:
            rgb_id = frontier["rgb_id"]
            feature = os.path.join(frontier_folder, rgb_id.replace(".png", "_full.pt"))
            stepdata["frontier_features"][rgb_id] = feature
        stepdata["snapshot_features"] = {}
        stepdata["snapshot_objects"] = {}
        snapshot_folder = os.path.join(epi_path, "object_features")
        for snapshot in stepdata["snapshots"]:
            rgb_id = snapshot["img_id"]
            feature = os.path.join(snapshot_folder, rgb_id.replace(".png", "_full.pt"))
            stepdata["snapshot_features"][rgb_id] = feature
            object_ids = snapshot["obj_ids"]
            # use image_id to index the snapshot
            stepdata["snapshot_objects"][rgb_id] = object_ids

        if stepdata["previous_choice"] is not None:
            stepdata["previous_choice"] = os.path.join(
                frontier_folder,
                stepdata["previous_choice"].replace(".png", "_full.pt"),
            )

        stepdata["egocentric_features"] = {}
        for view_idx in range(self.num_egocentric_views):
            egocentric_view_folder = os.path.join(epi_path, f"object_features")
            featrue = os.path.join(
                egocentric_view_folder, f"{step}-view_{view_idx}_full.pt"
            )
            stepdata["egocentric_features"][view_idx] = featrue
        return stepdata

    def load_data(self):
        # load scene feature into dict
        
        #1. load prefiltering candidates
        with open(self.ranking_path, "r") as f:
            self.candidate_rankings = json.load(f)
        #2. load augmented questions
        with open(self.augmented_questions_path, "r") as f:
            self.augmented_questions = json.load(f)
        self.obj_json_map = {}
        for obj_json in os.listdir(self.obj_bbox_dir):
            scene_id = obj_json.split(".")[0]
            self.obj_json_map[scene_id] = os.path.join(self.obj_bbox_dir, obj_json)

        self.episodes = []
        data = []
        self.episode2step = defaultdict(list)
        data_count = 0
        num_skipped = 0
        for i, episode in enumerate(os.listdir(self.explore_dir)):
            i -= num_skipped
            epi_path = os.path.join(self.explore_dir, episode)
            # load metadata
            try:
                with open(os.path.join(epi_path, "metadata.json"), "r") as f:
                    metadata = json.load(f)
            except:
                num_skipped += 1
                continue
            '''
            try:
                if (len(os.listdir(os.path.join(epi_path,"egocentric"))) == 0
                    or len(os.listdir(os.path.join(epi_path,"frontier_rgb"))) == 0
                    or len(os.listdir(os.path.join(epi_path,"object_features"))) == 0):
                    num_skipped += 1
                    continue
            except:
                num_skipped += 1
                continue
            '''
            self.episodes.append(metadata)
            # load step data
            steps_data = []
            target_obj_id = metadata["target_obj_id"]
            for step in range(metadata["episode_length"]):
                stepdata_path = os.path.join(epi_path, f"{pad_zero(str(step),4)}.json")
                if not os.path.exists(stepdata_path):
                    continue
                steps_data.append((stepdata_path, i, target_obj_id))
                self.episode2step[i].append(data_count)
                data_count += 1
            data.extend(steps_data)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        step_path, episode_id, target_obj_id = self.data[idx]
        # try:
        step = self.load_step(step_path)
        # except:
        #     index = np.random.choice(self.indices)
        #     return self.__getitem__(index)
        episode = self.episodes[episode_id]

        shuffle = self.random_permute and (self.split == "train")
        multi_src_features = []
        try:
            with open(self.obj_json_map[episode["scene"]]) as f:
                obj_json = json.load(f)
        except Exception as e:
            print(f"Error loading data at location {self.obj_json_map[episode['scene']]}: {e}")
            index = np.random.choice(self.indices)
            return self.__getitem__(index)
        # map scene_id to the object json directory
        obj_map = {obj["id"]: obj["class_name"] for obj in obj_json}
        obj_positions_map = {
            obj["id"]: (np.array(obj["bbox"][1]) + np.array(obj["bbox"][0])) / 2
            for obj in obj_json
        }
        try:
            obj_positions_map = {
                key: value[[0, 2, 1]] - step["position"]
                for key, value in obj_positions_map.items()
            }
        except:
            print("loss information in stepdata")
            print(step_path)
            index = np.random.choice(self.indices)
            return self.__getitem__(index)
        # format prefiltering
        prefilter_key = episode["question"] + "_" + episode["scene"]
        if "task_type" in episode.keys():
            prefilter_key += "_" + episode["task_type"]
        if prefilter_key in self.candidate_rankings.keys():
            ranking = self.candidate_rankings[prefilter_key]
        else:
            # skip sample without groundtruth ranking
            # ranking = list(set([obj["class_name"] for obj in obj_json])
            index = np.random.choice(self.indices)
            return self.__getitem__(index)
            
        question_text,question_feature = format_questions(episode, 
                False, self.augment_question, 
                self.augmented_questions, 
                self.image_prompt_visual_feature_size, 
                self.image_prompt_patch_size)
        if question_text is None:
            # the image prompt feature is not correctly loaded
            self.img_prompt_not_found_indices.add(idx)
            index = np.random.choice(self.indices)
            return self.__getitem__(index)
        text = question_text
        # add features for the image prompt
        multi_src_features.append(question_feature)
        if self.egocentric_views:
            try:
                egocentric_text, egocentric_features = prepare_egocentric_view(
                    step["egocentric_features"],
                    self.visual_feature_size,
                    self.patch_size,
                )
            except:
                index = np.random.choice(self.indices)
                return self.__getitem__(index)
            text += egocentric_text
            if self.add_positional_encodings:
                egocentric_positions = torch.cat(
                    [
                        torch.tensor(step["position"] - step["position"])
                        for _ in range(egocentric_features.shape[0])
                    ],
                    dim=0,
                )
                egocentric_features = sum_positional_encodings(
                    egocentric_features,
                    egocentric_positions,
                    self.positional_encoding,
                    num_bins=self.num_bins,
                    coord_range=self.bounds,
                )
            multi_src_features.append(egocentric_features)

        text += f"Select the frontier/snapshot that would help find the answer of the question.\n"

        if self.action_memory:
            try:
                memory_text, memory_feature = prepare_action_memory(
                    step["previous_choice"]
                )
            except:
                
                index = np.random.choice(self.indices)
                return self.__getitem__(index)
            text += memory_text
            multi_src_features.append(memory_feature)

        # replace scene graph in each steps with scene feature
        prediction = np.array(step["prediction"])
        snapshot_features, keep_indices = [], []
        snapshot_classes = []
        snapshot_positions = []
        snapshot_index = 0

        # prefiltering TODO
        # use seen objects set to replace class2object
        # class2object = defaultdict(list)
        seen_classes = set()
        use_gt = (random.random() < self.gt_rate) and self.mix_gt
        for i, rgb_id in enumerate(step["snapshot_features"].keys()):
            # No need to filter here (both scene_graph and snapshots objects from the json files)
            try:
                keep_indices.append(i)
                # a simple work round for step feature paths
                # snapshot_feature_path = step["snapshot_features"][rgb_id].split('/')
                # snapshot_feature_path[-1] = snapshot_feature_path[-1].replace('-','_')
                # snapshot_feature_path = '/'.join(snapshot_feature_path)
                snapshot_feature_path = step["snapshot_features"][rgb_id]
                '''
                snapshot_feature = torch.load(
                    step["snapshot_features"][rgb_id], map_location="cpu"
                )
                '''
                snapshot_feature = torch.load(
                    snapshot_feature_path, map_location="cpu"
                )
                #2*2*dim, 4 visual features to represent 1 snapshot
                snapshot_feature = merge_patches(
                    snapshot_feature.view(
                        self.visual_feature_size, self.visual_feature_size, -1
                    ),
                    self.patch_size,
                )
                # update the way naming objects
                '''
                if str(target_obj_id) in list(step["snapshot_objects"][rgb_id].keys()):
                    print('this is right')
                else:
                    print(target_obj_id)
                    print(list(step["snapshot_objects"][rgb_id].keys()))
                '''
                if use_gt:
                    snapshot_class = [
                        class_name['gt_class']
                        for sid,class_name in step["snapshot_objects"][rgb_id].items()
                    ]
                else:
                    snapshot_class = [
                        class_name['gt_class'] if sid == str(target_obj_id) and self.target_use_gt else class_name['recognize_class']
                        for sid,class_name in step["snapshot_objects"][rgb_id].items()
                    ]
                seen_classes.update(snapshot_class)
                snapshot_classes.append(
                    # [obj_map[str(sid)] for sid in step["snapshot_objects"][rgb_id]]
                    snapshot_class
                )
                snapshot_features.append(snapshot_feature)
                snapshot_positions.append(
                    torch.mean(
                        torch.tensor(
                            [
                                obj_positions_map[str(sid)]
                                for sid in step["snapshot_objects"][rgb_id]
                            ]
                        ),
                        dim=0,
                    )
                )
                snapshot_index += 1
            except Exception as e:
                # remove current wrong sample?
                if idx in set(self.indices):
                    self.indices = list(set(self.indices) - {idx})
                    #print(len(self.indices))
                    #print(f"Error loading data at location {step['snapshot_features'][rgb_id]}: {e}")
                index = np.random.choice(self.indices)
                return self.__getitem__(index)
        if self.add_positional_encodings:
            snapshot_features = [
                sum_positional_encodings(
                    snapshot_features[i].unsqueeze(0),
                    snapshot_positions[i],
                    self.positional_encoding,
                    num_bins=self.num_bins,
                    coord_range=self.bounds,
                ).squeeze(0)
                for i in range(len(snapshot_features))
            ]

        # Data Problem
        if not (
            np.where(prediction[keep_indices] == 1.0)[0].shape[0]
            + np.where(prediction[-len(step["frontiers"]) :] == 1.0)[0].shape[0]
            == 1
        ):
            assert "should not trigger this?", False
            self.obj_not_found_indices.add(idx)
            index = np.random.choice(self.indices)
            return self.__getitem__(index)

        # prefiltering TODO
        if self.prefiltering:
            # compute the intersection of ranking and class2object
            """
            print(episode["question"])
            print(seen_objects)
            print(ranking)
            """
            ranking = [cls for cls in ranking if cls in seen_classes]
            ranking = ranking[: self.top_k_categories]
            # 1. unlike objects, we can not directly order snapshots based on ranking, so
            # we simply filter out useless snapshots without reordering
            # keep snapshots that have at least one object in the ranking
            """
            print(f'top{self.top_k_categories} ranking', ranking)
            print("raw keep indices", keep_indices)
            print("raw snapshot classes", snapshot_classes)
            """
            ranking_set = set(ranking)
            snap_indices = [
                snap_idx
                for snap_idx in range(snapshot_index)
                if len(set(snapshot_classes[snap_idx]) & ranking_set) > 0
            ]
            keep_indices = [keep_indices[snap_idx] for snap_idx in snap_indices]
            snapshot_classes = [snapshot_classes[snap_idx] for snap_idx in snap_indices]
            # further filter out the useless objects (objects not in ranking) in each snapshot
            # print('before inner snapshot filtering', snapshot_classes)
            snapshot_classes = [
                [scls for scls in list(dict.fromkeys(snap_cls)) if scls in ranking_set]
                for snap_cls in snapshot_classes
            ]
            snapshot_features = [
                snapshot_features[snap_idx] for snap_idx in snap_indices
            ]
            snapshot_index = len(keep_indices)
        if shuffle:
            # shuffle the index if random_permute is True otherwise keep the original order
            random_snapshot_index = list(range(snapshot_index))
            np.random.shuffle(random_snapshot_index)
            keep_indices = [keep_indices[r_idx] for r_idx in random_snapshot_index]
            snapshot_classes = [
                snapshot_classes[r_idx] for r_idx in random_snapshot_index
            ]
            snapshot_features = [
                snapshot_features[r_idx] for r_idx in random_snapshot_index
            ]
        
        #text += "These are the snapshots (followed with contained object classes).\n"
        text 
        for i, class_names in enumerate(snapshot_classes):
            text += f"snapshot {i} "
            #text += f"{i} "
            class_names_set = set(class_names)
            class_names_list = list(class_names_set)
            sorted_class_names = sorted(class_names_list)
            '''
            if random.random() < self.mapping_rate:
                for class_name in sorted_class_names:
                    if self.map_category and class_name in self.category_map.keys():
                        #print(f"remap category {class_name} to {self.category_map[class_name]}")
                        text += f"{self.category_map[class_name]}, "
                    else:
                        text += f"{class_name}, "
            else:
            '''
            for class_name in sorted_class_names:
                text += f"{class_name}, "
            for _ in range(self.num_visual_tokens):
                text += "<scene>"
            text += " / "
        if snapshot_index == 0:
            text += f"No snapshot available "
            # construct zero scene feature if all snapshots are missed
            snapshot_features = None
        else:
            snapshot_features = torch.cat(snapshot_features, dim=0)
            multi_src_features.append(snapshot_features)

        text += "\n"

        frontier_index = list(range(len(step["frontiers"])))
        if shuffle:
            np.random.shuffle(frontier_index)
        frontier_text, frontier_features = prepare_frontier(
            step["frontier_features"],
            [step["frontiers"][idx] for idx in frontier_index],
            self.visual_feature_size,
            self.patch_size,
        )
        if frontier_text is None:
            index = np.random.choice(self.indices)
            return self.__getitem__(index)
        if self.add_positional_encodings:
            frontier_positions = torch.tensor(
                [step["frontier_positions"][idx] for idx in frontier_index]
            )
            frontier_features = sum_positional_encodings(
                frontier_features,
                frontier_positions,
                self.positional_encoding,
                num_bins=self.num_bins,
                coord_range=self.bounds,
            )

        text += frontier_text
        multi_src_features.append(frontier_features)
        # prefiltering TODO: move the first assertion here
        assert prediction.shape[0] == len(step["snapshot_features"]) + len(
            step["frontiers"]
        )
        # print("frontier prediction before shuffle", prediction[-len(step["frontiers"]):])
        prediction = np.concatenate(
            (
                prediction[keep_indices],
                prediction[
                    [idx + len(step["snapshot_features"]) for idx in frontier_index]
                ],
            )
        )
        prediction = torch.tensor(prediction)

        # prefiltering TODO: the assert might not be valid after prefiltering
        assert prediction.shape[0] == snapshot_index + len(step["frontiers"])

        if not np.where(prediction == 1.0)[0].shape[0] == 1:
            self.answer_obj_filtered_indices.add(idx)
            index = np.random.choice(self.indices)
            return self.__getitem__(index)

        prediction_index = np.where(prediction == 1.0)[0][0]
        if prediction_index < snapshot_index:
            answer = f"snapshot {prediction_index}"
        else:
            answer = f"frontier {prediction_index - snapshot_index}"

        text += "Answer: "
        text += answer + self.tokenizer.eos_token

        # randomly choose another item
        if snapshot_features is None and frontier_features is None:
            index = np.random.choice(self.indices)
            return self.__getitem__(index)

        multi_src_features = [f for f in multi_src_features if f is not None]
        scene_feature = torch.cat(multi_src_features, dim=0)
        # print("scene_feature", scene_feature.shape)

        
        if len(scene_feature) // self.num_visual_tokens > 45:
            self.too_many_objects_indices.add(idx)
            if self.split == "train":
                index = np.random.choice(self.indices)
                return self.__getitem__(index)
        
        step["scene_feature"] = scene_feature
        
        
        if self.max_length <= len(text):
            self.too_long_prompts_indices.add(idx)
            #print(f"the number of visual tokens for each item {self.num_visual_tokens}")
            #print(len(scene_feature))
            #print(text)
            if self.split == "train":
                index = np.random.choice(self.indices)
                return self.__getitem__(index)
        
        # assert self.max_length > len(text)
        # assert self.max_length > len(
        #     scene_feature
        # )  # make sure that scene feature is never truncated

        text = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = text["input_ids"]
        length = torch.nonzero(input_ids).shape[0]
        decode_result = self.tokenizer.decode(input_ids[0][0:length])
        if "<unk>" in decode_result:
            print("unknow problem in tokenizer!")

        attention_mask = text["attention_mask"]

        scene_insert_loc = (
            (input_ids == self.scene_token_id).nonzero()[:, 1].reshape(-1)
        )
        input_dict = EasyDict(
            text=text,
            input_ids=input_ids,
            length=length,
            scene_length=len(scene_feature),
            attention_mask=attention_mask,
            scene_feature=scene_feature,
            scene_insert_loc=scene_insert_loc,
        )
        # add prompt input for prefiltering
        if self.prefiltering:
            classes = list(seen_classes)
            if shuffle:
                np.random.shuffle(classes)
            (
                input_dict.filter_input_ids,
                input_dict.filter_length,
                input_dict.filter_attention_mask,
            ) = prepare_prefiltering_input(
                question_text,
                self.tokenizer,
                classes,
                ranking,
                self.max_length,
                self.top_k_categories,
            )
            # prepare image prompt feature for prefiltering
            if question_feature is not None:
                input_dict.filter_feature = question_feature
                input_dict.filter_insert_loc = (
                    (input_dict.filter_input_ids == self.scene_token_id).nonzero()[:, 1].reshape(-1)
                )
            else:
                input_dict.filter_feature = torch.empty((0,1024))
                input_dict.filter_insert_loc = torch.empty((0,))
        return input_dict

    def collate_wrapper(self, batch):
        # because sos token is added, the max_length should be +1?
        max_length = max(b.length for b in batch) + 1
        max_scene_length = max(b.scene_feature.shape[0] for b in batch)

        scene_feature = torch.zeros((len(batch), max_scene_length, 1024))
        scene_insert_loc = torch.zeros((len(batch), max_scene_length))

        for j, b in enumerate(batch):
            scene_feature[j, : b.scene_feature.shape[0]] = b.scene_feature
            scene_insert_loc[j, : b.scene_insert_loc.shape[0]] = b.scene_insert_loc

        if self.prefiltering:
            max_filter_length = max(b.filter_length for b in batch) + 1
            max_filter_feature_length = max(b.filter_feature.shape[0] for b in batch)
            
            # align filter feature 
            filter_feature = torch.zeros((len(batch), max_filter_feature_length, 1024))
            filter_insert_loc = torch.zeros((len(batch), max_filter_feature_length))
            
            for j, b in enumerate(batch):
                filter_feature[j, : b.filter_feature.shape[0]] = b.filter_feature
                filter_insert_loc[j, : b.filter_insert_loc.shape[0]] = b.filter_insert_loc
                
            return EasyDict(
                input_ids=torch.cat([b.input_ids for b in batch])[..., :max_length],
                attention_mask=torch.cat([b.attention_mask for b in batch])[
                    ..., :max_length
                ],
                scene_feature=scene_feature,
                scene_insert_loc=scene_insert_loc.to(torch.long),
                scene_length=torch.tensor([b.scene_length for b in batch]),
                max_scene_length=torch.tensor(
                    [b.scene_feature.shape[0] for b in batch]
                ),
                filter_input_ids=torch.cat([b.filter_input_ids for b in batch])[
                    ..., :max_filter_length
                ],
                filter_attention_mask=torch.cat(
                    [b.filter_attention_mask for b in batch]
                )[..., :max_filter_length],
                filter_feature=filter_feature,
                filter_insert_loc=filter_insert_loc.to(torch.long),
                filter_feature_length=torch.tensor([b.filter_feature.shape[0] for b in batch]),
                max_filter_feature_length=torch.tensor(
                    [b.scene_feature.shape[0] for b in batch]
                )
            )
        return EasyDict(
            input_ids=torch.cat([b.input_ids for b in batch])[..., :max_length],
            attention_mask=torch.cat([b.attention_mask for b in batch])[
                ..., :max_length
            ],
            scene_feature=scene_feature,
            scene_insert_loc=scene_insert_loc.to(torch.long),
            scene_length=torch.tensor([b.scene_length for b in batch]),
            max_scene_length=torch.tensor([b.scene_feature.shape[0] for b in batch]),
        )

    def split_index(self, test_ratio=0.3):
        test_num = int(test_ratio * len(self.episodes))
        test_episode = random.sample(range(len(self.episodes)), test_num)
        test_episode = [
            i
            for i in range(len(self.episodes))
            if int(self.episodes[i]["scene"].split("-")[0]) > 749
            and int(self.episodes[i]["scene"].split("-")[0]) < 900
        ]
        train_episode = [
            i
            for i in range(len(self.episodes))
            if int(self.episodes[i]["scene"].split("-")[0]) < 749
        ]
        train_index, test_index = [], []
        for i in self.episode2step.keys():
            if i in test_episode:
                test_index.extend(self.episode2step[i])
            if i in train_episode:
                train_index.extend(self.episode2step[i])
        return train_index, test_index