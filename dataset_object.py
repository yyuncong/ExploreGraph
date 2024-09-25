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


def prepare_egocentric_view(egocentric_path, num_egocentric_views,
        visual_feature_size, patch_size):
    text = "Followings are the egocentric views:\n "
    num_tokens = (visual_feature_size // patch_size) ** 2
    egocentric_features = []
    # TODO: only pick the last egocentric view if num_egocentric_views == 1
    egocentric_path = sorted(list(egocentric_path.items()), key = lambda x: x[0])
    #print(egocentric_path)
    if num_egocentric_views == 1:
        egocentric_path = [egocentric_path[-1]]
        
    for i, view in egocentric_path:
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
        text = f"Below are all the frontiers that we can explore:\n"
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
        augment_question=False,
        top_k_categories=5,
        num_egocentric_views=5,
        patch_size=3,
        visual_feature_size=6,
        egocentric_patch_size=2,
        egocentric_visual_size=24,
        split="train",
    ):
        self.ranking_path = os.path.join(scene_path, "selected_candidates.json")
        #self.obj_bbox_dir = "/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d/hm3d_obj_bbox_merged"
        self.obj_bbox_dir ="/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d/hm3d_obj_bbox_all"
        self.explore_dir = os.path.join(exploration_path, "exploration_data_cg_baseline")
        #self.explore_dir = os.path.join(exploration_path, "exploration_data")
        self.augmented_questions_path = "question_augment/augmented_generated_questions.json"
        self.tokenizer = tokenizer
        self.scene_token = scene_token
        self.scene_token_id = self.tokenizer(self.scene_token).input_ids[-1]
        self.egocentric_views = egocentric_views
        self.action_memory = action_memory
        self.prefiltering = prefiltering
        self.random_permute = random_permute
        self.augment_question = augment_question
        self.num_egocentric_views = num_egocentric_views
        self.egocentric_patch_size = egocentric_patch_size
        self.egocentric_visual_size = egocentric_visual_size
        self.top_k_categories = top_k_categories

        self.max_length = max_length
        self.split = split
        self.data = self.load_data()

        train_index, test_index = self.split_index()
        self.indices = train_index if split == "train" else test_index
        self.obj_not_found_indices = set({})
        self.too_many_objects_indices = set({})
        self.too_long_prompts_indices = set({})
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
        stepdata["object_features"] = {}
        #stepdata["object_ids"] = {}
        # TODO: modify it to object baseline setting
        object_folder = os.path.join(epi_path, "image_crop")
        for obj in stepdata["objects"]:
            rgb_id = obj["img_id"]
            feature = os.path.join(object_folder, rgb_id.replace(".png", "_full.pt"))
            stepdata["object_features"][rgb_id] = feature
            #object_id = snapshot["obj_id"]
            # use image_id to index the snapshot
            #stepdata["object_ids"][rgb_id] = object_ids

        if stepdata["previous_choice"] is not None:
            stepdata["previous_choice"] = os.path.join(
                frontier_folder,
                stepdata["previous_choice"].replace(".png", "_full.pt"),
            )

        stepdata["egocentric_features"] = {}
        # TODO: load all egocentric views for current step
        egocentric_view_folder = os.path.join(epi_path, f"egocentric_views")
        step_prefix, step_suffix = f"{step}-view_","_full.pt"
        #print(step)
        #print(os.listdir(egocentric_view_folder))
        egocentric_files = sorted([f for f in os.listdir(egocentric_view_folder) if (step_prefix in f and step_suffix in f)])
        #print(egocentric_files)
        #exit(0)
        for f in egocentric_files:
            view_idx = int(f.split('_')[1])
            stepdata["egocentric_features"][view_idx] = os.path.join(egocentric_view_folder,f)
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
        ranking = self.candidate_rankings[episode["question"] + "_" + episode["scene"]]
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
            

        if self.augment_question:
            raw_question = episode["question"]
            if episode["question"] in self.augmented_questions.keys():
                phrased_question = np.random.choice(self.augmented_questions[episode["question"]])
                #print(f"raw question {raw_question} phrased question {phrased_question}")
            else:
                phrased_question = raw_question
            text = f"Question: {phrased_question}\n"
        else:
            text = f"Question: {episode['question']}\n"

        if self.egocentric_views:
            try:
                egocentric_text, egocentric_features = prepare_egocentric_view(
                    step["egocentric_features"],
                    self.num_egocentric_views,
                    self.egocentric_visual_size,
                    self.egocentric_patch_size,
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

        text += f"Select the frontier/object that would help finding the answer of the question.\n"

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
        object_features, keep_indices = [], []
        object_classes = []
        object_positions = []
        object_index = 0

        # prefiltering TODO
        # use seen objects set to replace class2object
        # class2object = defaultdict(list)
        seen_classes = set()
        for i, rgb_id in enumerate(step["object_features"].keys()):
            # No need to filter here (both scene_graph and snapshots objects from the json files)
            try:
                keep_indices.append(i)
                object_feature_path = step["object_features"][rgb_id]
                object_feature = torch.load(
                    object_feature_path, map_location="cpu"
                )
                #2*2*dim, 4 visual features to represent 1 snapshot
                object_feature = merge_patches(
                    object_feature.view(
                        self.visual_feature_size, self.visual_feature_size, -1
                    ),
                    self.patch_size,
                )
                # update the way naming objects
                object_info = step["objects"][i]
                object_class = object_info["recognize_class"]
                seen_classes.add(object_class)
                object_classes.append(object_class)
                object_features.append(object_feature)
                object_positions.append(
                    torch.tensor(
                        obj_positions_map[str(object_info['obj_id'])]
                    )
                )
                object_index += 1
            except Exception as e:
                # remove current wrong sample?
                if idx in set(self.indices):
                    self.indices = list(set(self.indices) - {idx})
                    #print(len(self.indices))
                    #print(f"Error loading data at location {step['snapshot_features'][rgb_id]}: {e}")
                index = np.random.choice(self.indices)
                return self.__getitem__(index)
        if self.add_positional_encodings:
            object_features = [
                sum_positional_encodings(
                    object_features[i].unsqueeze(0),
                    object_positions[i],
                    self.positional_encoding,
                    num_bins=self.num_bins,
                    coord_range=self.bounds,
                ).squeeze(0)
                for i in range(len(object_features))
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
            
            object_indices = [
                obj_idx for obj_idx in range(object_index)
                if object_classes[obj_idx] in ranking_set
            ]
            keep_indices = [keep_indices[obj_idx] for obj_idx in object_indices]
            object_classes = [object_classes[obj_idx] for obj_idx in object_indices]
            # further filter out the useless objects (objects not in ranking) in each snapshot
            # print('before inner snapshot filtering', snapshot_classes)
            object_features = [
                object_features[obj_idx] for obj_idx in object_indices
            ]
            object_index = len(keep_indices)
        if shuffle:
            # shuffle the index if random_permute is True otherwise keep the original order
            random_object_index = list(range(object_index))
            np.random.shuffle(random_object_index)
            keep_indices = [keep_indices[r_idx] for r_idx in random_object_index]
            object_classes = [
                object_classes[r_idx] for r_idx in random_object_index
            ]
            object_features = [
                object_features[r_idx] for r_idx in random_object_index
            ]
        
        text += "These are the objects:\n"
        for i, class_name in enumerate(object_classes):
            text += f"object {i} {class_name}"
            #text += f"{i} "
            '''
            class_names_set = set(class_names)
            class_names_list = list(class_names_set)
            sorted_class_names = sorted(class_names_list)
            for class_name in sorted_class_names:
                text += f"{class_name}, "
            '''
            for _ in range(self.num_visual_tokens):
                text += "<scene>"
            text += " / "
        if object_index == 0:
            text += f"No object available "
            # construct zero scene feature if all snapshots are missed
            object_features = None
        else:
            object_features = torch.cat(object_features, dim=0)
            multi_src_features.append(object_features)

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
        assert prediction.shape[0] == len(step["object_features"]) + len(
            step["frontiers"]
        )
        # print("frontier prediction before shuffle", prediction[-len(step["frontiers"]):])
        prediction = np.concatenate(
            (
                prediction[keep_indices],
                prediction[
                    [idx + len(step["object_features"]) for idx in frontier_index]
                ],
            )
        )
        prediction = torch.tensor(prediction)

        # prefiltering TODO: the assert might not be valid after prefiltering
        assert prediction.shape[0] == object_index + len(step["frontiers"])

        if not np.where(prediction == 1.0)[0].shape[0] == 1:
            self.answer_obj_filtered_indices.add(idx)
            index = np.random.choice(self.indices)
            return self.__getitem__(index)

        prediction_index = np.where(prediction == 1.0)[0][0]
        if prediction_index < object_index:
            answer = f"object {prediction_index}"
        else:
            answer = f"frontier {prediction_index - object_index}"

        text += "Answer: "
        text += answer + self.tokenizer.eos_token

        # randomly choose another item
        if object_features is None and frontier_features is None:
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
                episode["question"],
                self.tokenizer,
                classes,
                ranking,
                self.max_length,
                self.top_k_categories,
            )
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
            if int(self.episodes[i]["scene"].split("-")[0]) > 880
            and int(self.episodes[i]["scene"].split("-")[0]) < 900
        ]
        train_episode = [
            i
            for i in range(len(self.episodes))
            if int(self.episodes[i]["scene"].split("-")[0]) < 800
        ]
        train_index, test_index = [], []
        for i in self.episode2step.keys():
            if i in test_episode:
                test_index.extend(self.episode2step[i])
            if i in train_episode:
                train_index.extend(self.episode2step[i])
        return train_index, test_index