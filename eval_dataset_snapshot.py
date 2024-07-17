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


def prepare_egocentric_view(egocentric_path):
    text = "Followings are the egocentric views:\n "
    egocentric_features = []
    for i, view in egocentric_path.items():
        egocentric_features.append(torch.load(view, map_location="cpu"))
        text += f"<scene> "
    egocentric_features = torch.cat(egocentric_features, dim=0)
    text += "/\n"
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


def prepare_frontier(feature_path, frontier_info):
    # print("frontier after shuffle", [info['rgb_id'] for info in frontier_info])
    try:
        text = f"Below are all the frontiers that we can explore:\n"
        if len(frontier_info) > 0:
            frontier_features = []
            for i, info in enumerate(frontier_info):
                frontier_features.append(
                    torch.load(feature_path[info["rgb_id"]], map_location="cpu")
                )
                text += f"frontier {i} <scene> "
            frontier_features = torch.cat(frontier_features, dim=0)
        else:
            text += f"No frontier available "
            frontier_features = None
        text += "/\n"
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
    # print("raw text of filter prompt:", filter_text)
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


# format object input after generating prefiltering result
def prepare_snapshot_input(
    seen_classes,
    snapshot_prediction,
    snapshot_classes,
    snapshot_features,
    prefiltering,
    ranking,
    topk,
):

    # Cases where prefiltering over objects is needed
    # revise object processing
    snapshot_index = len(snapshot_classes)
    if prefiltering:
        #print("seen objects", seen_classes)
        ranking = [cls for cls in ranking if cls in seen_classes]
        ranking = ranking[:topk]
        ranking_set = set(ranking)
        snap_indices = [
            snap_idx
            for snap_idx in range(snapshot_index)
            if len(set(snapshot_classes[snap_idx]) & ranking_set) > 0
        ]
        #print("raw prediction", snapshot_prediction)
        #print("raw classes", snapshot_classes)
        #print("filtered indices", snap_indices)
        snapshot_prediction = [
            snapshot_prediction[snap_idx]
            for snap_idx in snap_indices
        ]
        snapshot_classes = [
            snapshot_classes[snap_idx]
            for snap_idx in snap_indices
        ]
        #print('before inner snapshot filtering', snapshot_classes)
        snapshot_classes = [
                [scls for scls in list(dict.fromkeys(snap_cls)) if scls in ranking_set] 
                for snap_cls in snapshot_classes
        ]
        #print('after inner snapshot filtering', snapshot_classes)
        snapshot_features = [
            snapshot_features[snap_idx]
            for snap_idx in snap_indices
        ]
        #print("filtered prediction", snapshot_prediction)
        #print("filtered classes", snapshot_classes)
        snapshot_index = len(snap_indices)

    text = "These are the snapshots:\n"
    for i, class_names in enumerate(snapshot_classes):
        text += f"snapshot {i} <scene> "
        #class_names_set = set(class_names)
        for class_name in class_names:
            text += f"{class_name}, "
        

    if snapshot_index == 0:
        text += f"No snapshot available "
        # construct zero scene feature if all objects are missed
        snapshot_features = None
    else:
        snapshot_features = torch.cat(snapshot_features, dim=0)
    text += "/\n"
    # print("object prompt \n", text)
    return text, snapshot_features, snapshot_prediction, snapshot_index


def construct_selection_prompt(
    tokenizer,
    scene_token_id,
    text_before_snapshot,
    feature_before_snapshot,
    frontier_text,
    frontier_features,
    frontier_prediction,
    # dict object contains object features/predictions/classes
    snapshot_info_dict,
    prefiltering,
    # parse result of prefiltering output
    ranking,
    topk,
    max_length,
):
    snapshot_text, snapshot_features, snapshot_prediction, snapshot_index = (
        prepare_snapshot_input(
            snapshot_info_dict.seen_classes,
            snapshot_info_dict.prediction,
            snapshot_info_dict.classes,
            snapshot_info_dict.features,
            prefiltering,
            ranking,
            topk,
        )
    )

    text = text_before_snapshot + snapshot_text + frontier_text
    # format scene feature
    if snapshot_features is None and frontier_features is None:
        return "missing features"

    scene_feature = feature_before_snapshot + [snapshot_features] + [frontier_features]
    scene_feature = [f for f in scene_feature if f is not None]
    scene_feature = torch.cat(scene_feature, dim=0)
    if len(scene_feature) > 50:
        return "too many objects"

    # format prediction
    prediction = np.concatenate((snapshot_prediction, frontier_prediction))
    prediction = torch.tensor(prediction)
    assert prediction.shape[0] == snapshot_index + len(frontier_prediction)

    # This means the prefiltering result is incorrect
    if not np.where(prediction == 1.0)[0].shape[0] == 1:
        return "incorrect prefiltering"

    prediction_index = np.where(prediction == 1.0)[0][0]
    if prediction_index < snapshot_index:
        answer = f"snapshot {prediction_index}"
    else:
        answer = f"frontier {prediction_index - snapshot_index}"

    # format answer
    text += "Answer: "
    text += answer + tokenizer.eos_token
    # print("final selection prompt \n", text)
    if max_length <= len(text):
        return "input too long"

    text = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    input_ids = text["input_ids"]
    length = torch.nonzero(input_ids).shape[0]

    attention_mask = text["attention_mask"]

    scene_insert_loc = (input_ids == scene_token_id).nonzero()[:, 1].reshape(-1)

    input_dict = EasyDict(
        text=text,
        input_ids=input_ids,
        length=length,
        scene_length=len(scene_feature),
        attention_mask=attention_mask,
        scene_feature=scene_feature,
        scene_insert_loc=scene_insert_loc,
    )
    return input_dict


# separate collection function for stage 2 selection prompt
def collate_selection_wrapper(batch):

    # because sos token is added, the max_length should be +1?
    max_length = max(b.length for b in batch) + 1
    max_scene_length = max(b.scene_feature.shape[0] for b in batch)
    # max_frontier_length = max(b.frontier_feature.shape[0] for b in batch)

    scene_feature = torch.zeros((len(batch), max_scene_length, 1024))
    scene_insert_loc = torch.zeros((len(batch), max_scene_length))

    for j, b in enumerate(batch):
        scene_feature[j, : b.scene_feature.shape[0]] = b.scene_feature
        # frontier_feature[j, :b.frontier_feature.shape[0]] = b.frontier_feature
        scene_insert_loc[j, : b.scene_insert_loc.shape[0]] = b.scene_insert_loc

    return EasyDict(
        input_ids=torch.cat([b.input_ids for b in batch])[..., :max_length],
        attention_mask=torch.cat([b.attention_mask for b in batch])[..., :max_length],
        scene_feature=scene_feature,
        scene_insert_loc=scene_insert_loc.to(torch.long),
        scene_length=torch.tensor([b.scene_length for b in batch]),
        max_scene_length=torch.tensor([b.scene_feature.shape[0] for b in batch]),
    )


class ExploreDataset(Dataset):
    def __init__(
        self,
        scene_path,
        exploration_path,
        tokenizer,
        max_length,
        scene_token=SCENE_TOKEN,
        # frontier_token = FRONTIER_TOKEN,
        select_token=SELECT_TOKEN,
        egocentric_views=False,
        action_memory=False,
        prefiltering=False,
        random_permute=False,
        add_positional_encodings=False,
        # Jiachen TODO: add your parameter here
        top_k_categories=5,
        num_egocentric_views=5,
        split="train",
    ):
        # scene_path = "/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory"
        self.scene_dir = os.path.join(scene_path, "scene_feature_dict_merged_snapshots")
        self.ranking_path = os.path.join(scene_path, "selected_candidates.json")
        # exploration_path = (
        #     "/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/3d/explore-eqa-test/"
        # )
        self.obj_bbox_dir = "/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d/hm3d_obj_bbox_merged"
        self.explore_dir = os.path.join(exploration_path, "exploration_data_new_new")
        self.tokenizer = tokenizer
        self.scene_token = scene_token
        self.scene_token_id = self.tokenizer(self.scene_token).input_ids[-1]
        self.egocentric_views = egocentric_views
        self.action_memory = action_memory
        self.prefiltering = prefiltering
        self.random_permute = random_permute
        self.num_egocentric_views = num_egocentric_views
        self.top_k_categories = top_k_categories

        # self.frontier_token = frontier_token
        # self.frontier_token_id = self.tokenizer.convert_tokens_to_ids(self.frontier_token)
        # self.select_token = select_token
        # self.select_token_id = self.tokenizer(select_token).input_ids[-1]
        self.max_length = max_length
        self.split = split
        self.data = self.load_data()

        train_index, test_index = self.split_index()
        self.indices = train_index if split == "train" else test_index
        self.obj_not_found_indices = set({})
        self.too_many_objects_indices = set({})
        self.answer_obj_filtered_indices = set({})
        self.bounds = (-7, 7)
        self.num_bins = 128
        self.positional_encoding = positionalencoding2d(
            1024, self.num_bins, self.num_bins
        )
        self.add_positional_encodings = add_positional_encodings

    def load_step(self, step_path):
        with open(step_path, "r") as f:
            stepdata = json.load(f)
        epi_path = "/".join(step_path.split("/")[:-1])
        step_file_name = step_path.split("/")[-1]
        step = int(step_file_name.split(".")[0])

        # add paths for frontiers
        stepdata["frontier_features"] = {}
        stepdata["position"] = np.array(stepdata["agent_state"]["init_pts"])[None,]
        stepdata["frontier_positions"] = (
            np.array([f["coordinate"] for f in stepdata["frontiers"]])
            - stepdata["position"]
        )
        frontier_folder = os.path.join(epi_path, "frontier_rgb")
        for frontier in stepdata["frontiers"]:
            # placeholder for loading frontier feature
            rgb_id = frontier["rgb_id"]
            feature = os.path.join(frontier_folder, rgb_id.replace(".png", ".pt"))
            stepdata["frontier_features"][rgb_id] = feature
        
        stepdata["snapshot_features"] = {}
        stepdata["snapshot_objects"] = {}
        snapshot_folder = os.path.join(epi_path, "object_features")
        for snapshot in stepdata["snapshots"]:
            rgb_id = snapshot["img_id"]
            feature = os.path.join(snapshot_folder, rgb_id.replace(".png", ".pt"))
            stepdata["snapshot_features"][rgb_id] = feature
            object_ids = snapshot["obj_ids"]
            stepdata["snapshot_objects"][rgb_id] = object_ids
            
        if stepdata["previous_choice"] is not None:
            stepdata["previous_choice"] = os.path.join(
                frontier_folder,
                stepdata["previous_choice"].replace(".png", ".pt"),
            )
            
        stepdata["egocentric_features"] = {}
        for view_idx in range(self.num_egocentric_views):
            egocentric_view_folder = os.path.join(epi_path, f"egocentric")
            featrue = os.path.join(egocentric_view_folder, f"{step}_view_{view_idx}.pt")
            stepdata["egocentric_features"][view_idx] = featrue
        return stepdata

    def load_data(self):

        # Jiachen TODO: load your "question/scene to ranking json" here

        # load scene feature into dict
        with open(self.ranking_path, "r") as f:
            self.candidate_rankings = json.load(f)
    
        self.obj_json_map = {}
        for obj_json in os.listdir(self.obj_bbox_dir):
            scene_id = obj_json.split(".")[0]
            self.obj_json_map[scene_id] = os.path.join(self.obj_bbox_dir, obj_json)

        # load episode data: metadata is managed with self.episodes
        # TODO later: Remove num skipped to improve error handling
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
            self.episodes.append(metadata)

            # load step data
            steps_data = []
            for step in range(metadata["episode_length"]):
                stepdata_path = os.path.join(epi_path, f"{pad_zero(str(step),4)}.json")
                steps_data.append((stepdata_path, i))
                self.episode2step[i].append(data_count)
                data_count += 1
            data.extend(steps_data)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Jiachen TODO: add your feature to get item
        # 0 no need to shuffle here
        # 1 format prefiltering prompt and answer
        # 1.1 load data and prepare the input for prefiltering: ranking/seen object categories
        # 1.2 prepare prompt before object: egocentric view and action memory
        # 1.3 prepare prompt after object: frontiers and expected answer
        # 2 parse the generation result of model: which classes are chosen?
        # 3 format the selection prompt
        # 4 parse the selection result and evaluate the result

        # try:
        # load a whole episode and each step within it
        step_path, episode_id = self.data[idx]
        try:
            step = self.load_step(step_path)
        
        except:
            index = np.random.choice(self.indices)
            return self.__getitem__(index)
        
        episode = self.episodes[episode_id]
        # shuffle = self.random_permute and (self.split == "train")
        # Jiachen TODO 1: load ranking
        ranking = self.candidate_rankings[episode["question"] + "_" + episode["scene"]]
        # collections of features from egocentric view/action memory/scene graph/frontiers
        feature_before_snapshot = []
        # separately consider text before object (Questions, egocentric views, action memory)
        # text after object (frontiers, answer)d
        # text_before_object = ""

        with open(self.obj_json_map[episode["scene"]]) as f:
            obj_json = json.load(f)
        obj_map = {obj["id"]: obj["class_name"] for obj in obj_json}
        obj_positions_map = {
            obj["id"]: (np.array(obj["bbox"][1]) + np.array(obj["bbox"][0])) / 2
            for obj in obj_json
        }
        obj_positions_map = {
            key: value[[0, 2, 1]] - step["position"]
            for key, value in obj_positions_map.items()
        }
        #print('Successfully loaded basic files!')
        text_before_snapshot = f"Question: {episode['question']}\n"

        if self.egocentric_views:
            try:
                egocentric_text, egocentric_features = prepare_egocentric_view(
                    step["egocentric_features"]
                )
            except:
                index = np.random.choice(self.indices)
                return self.__getitem__(index)
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

            text_before_snapshot += egocentric_text
            feature_before_snapshot.append(egocentric_features)

        text_before_snapshot += f"Select the frontier/snapshot that would help finding the answer of the question.\n"

        if self.action_memory:
            memory_text, memory_feature = prepare_action_memory(step["previous_choice"])
            text_before_snapshot += memory_text
            feature_before_snapshot.append(memory_feature)

        # replace scene graph in each steps with scene feature
        # Jiachen TODO 2: extract seen object categories at the same time
        prediction = np.array(step["prediction"])
        snapshot_features, snapshot_classes, keep_indices = [], [], []
        snapshot_positions = []
        snapshot_index = 0
        seen_classes = set()
        for i, rgb_id in enumerate(step["snapshot_features"]):
            try: 
                keep_indices.append(i)
                snapshot_feature = torch.load(
                    step["snapshot_features"][rgb_id], map_location="cpu"
                )
                snapshot_class = [obj_map[str(sid)] for sid in step["snapshot_objects"][rgb_id]]
                seen_classes.update(snapshot_class)
                snapshot_classes.append(
                    #[obj_map[str(sid)] for sid in step["snapshot_objects"][rgb_id]]
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
            except:
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
        snapshot_prediction = prediction[keep_indices]
        frontier_prediction = prediction[-len(step["frontiers"]) :]
        if not (
            np.where(snapshot_prediction == 1.0)[0].shape[0]
            + np.where(frontier_prediction == 1.0)[0].shape[0]
            == 1
        ):
            assert "should not trigger this?", False
            self.obj_not_found_indices.add(idx)
            index = np.random.choice(self.indices)
            return self.__getitem__(index)

        # This is the target ranking for prefiltering
        ranking = [cls for cls in ranking if cls in seen_classes]
        ranking = ranking[: self.top_k_categories]
        # print("the list of seen objects", list(class2object.keys()))
        # print(f"the top {self.top_k_categories} ranking of seen objects", ranking)
        """
        object_text, object_features, object_prediction = prepare_object_input(
            class2object,
            object_prediction,
            object_classes,
            object_features,
            self.prefiltering,
            ranking,
            self.top_k_categories,
        )
        """
        frontier_index = list(range(len(step["frontiers"])))
        frontier_text, frontier_features = prepare_frontier(
            step["frontier_features"],
            [step["frontiers"][idx] for idx in frontier_index],
        )
        frontier_positions = torch.tensor(step["frontier_positions"])
        if self.add_positional_encodings:
            frontier_features = sum_positional_encodings(
                frontier_features,
                frontier_positions,
                self.positional_encoding,
                num_bins=self.num_bins,
                coord_range=self.bounds,
            )
        #print('frontier_text', frontier_text)
        if frontier_text is None:
            index = np.random.choice(self.indices)
            return self.__getitem__(index)
        # text += frontier_text
        # add frontier features
        # multi_src_features.append(frontier_features)
        # print("prediction before reformat", prediction)
        # prepare prediction and answer
        # add prompt input for prefiltering
        # assume always use prefiltering
        # if self.prefiltering:
        input_dict = EasyDict()
        classes = list(seen_classes)
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
        selection_dict = EasyDict(
            scene_token_id=self.scene_token_id,
            text_before_snapshot=text_before_snapshot,
            feature_before_snapshot=feature_before_snapshot,
            frontier_text=frontier_text,
            frontier_features=frontier_features,
            frontier_prediction=frontier_prediction,
            snapshot_info_dict=EasyDict(
                seen_classes=seen_classes,
                prediction=snapshot_prediction,
                classes=snapshot_classes,
                features=snapshot_features,
            ),
            prefiltering=self.prefiltering,
            ranking=ranking,
            topk=self.top_k_categories,
        )
        input_dict.selection_dict = selection_dict
        return input_dict

    # def collate_wrapper(self, batch):
    #     # because sos token is added, the max_length should be +1?
    #     max_length = max(b.length for b in batch) + 1
    #     max_scene_length = max(b.scene_feature.shape[0] for b in batch)
    #     # max_frontier_length = max(b.frontier_feature.shape[0] for b in batch)

    #     scene_feature = torch.zeros((len(batch), max_scene_length, 1024))
    #     scene_insert_loc = torch.zeros((len(batch), max_scene_length))

    #     for j, b in enumerate(batch):
    #         scene_feature[j, : b.scene_feature.shape[0]] = b.scene_feature
    #         # frontier_feature[j, :b.frontier_feature.shape[0]] = b.frontier_feature
    #         scene_insert_loc[j, : b.scene_insert_loc.shape[0]] = b.scene_insert_loc

    #     if self.prefiltering:
    #         max_filter_length = max(b.filter_length for b in batch) + 1
    #         return EasyDict(
    #             input_ids=torch.cat([b.input_ids for b in batch])[..., :max_length],
    #             attention_mask=torch.cat([b.attention_mask for b in batch])[
    #                 ..., :max_length
    #             ],
    #             scene_feature=scene_feature,
    #             scene_insert_loc=scene_insert_loc.to(torch.long),
    #             scene_length=torch.tensor([b.scene_length for b in batch]),
    #             max_scene_length=torch.tensor(
    #                 [b.scene_feature.shape[0] for b in batch]
    #             ),
    #             # Jiachen TODO 7
    #             filter_input_ids=torch.cat([b.filter_input_ids for b in batch])[
    #                 ..., :max_filter_length
    #             ],
    #             filter_attention_mask=torch.cat(
    #                 [b.filter_attention_mask for b in batch]
    #             )[..., :max_filter_length],
    #         )
    #     return EasyDict(
    #         input_ids=torch.cat([b.input_ids for b in batch])[..., :max_length],
    #         attention_mask=torch.cat([b.attention_mask for b in batch])[
    #             ..., :max_length
    #         ],
    #         scene_feature=scene_feature,
    #         scene_insert_loc=scene_insert_loc.to(torch.long),
    #         scene_length=torch.tensor([b.scene_length for b in batch]),
    #         max_scene_length=torch.tensor([b.scene_feature.shape[0] for b in batch]),
    #     )

    def collate_wrapper(self, batch):
        # wrap up the prefiltering batch
        max_filter_length = max(b.filter_length for b in batch) + 1
        return EasyDict(
            # Jiachen TODO 7
            filter_input_ids=torch.cat([b.filter_input_ids for b in batch])[
                ..., :max_filter_length
            ],
            filter_attention_mask=torch.cat([b.filter_attention_mask for b in batch])[
                ..., :max_filter_length
            ],
            filter_length=torch.tensor([b.filter_length for b in batch]),
            # dummy wrapper for selection prompt
            selection_dict=[b.selection_dict for b in batch],
        )

    # split the dataset by episode id
    def split_index(self, test_ratio=0.3):
        test_num = int(test_ratio * len(self.episodes))
        test_episode = random.sample(range(len(self.episodes)), test_num)
        test_episode = [
            i
            for i in range(len(self.episodes))
            if int(self.episodes[i]["scene"].split("-")[0]) > 700
            and int(self.episodes[i]["scene"].split("-")[0]) < 730
        ]
        train_episode = [
            i
            for i in range(len(self.episodes))
            if int(self.episodes[i]["scene"].split("-")[0]) <= 700
        ]
        # print("test episode", test_episode)
        train_index, test_index = [], []
        # print(self.episode2step)
        for i in self.episode2step.keys():
            if i in test_episode:
                test_index.extend(self.episode2step[i])
            if i in train_episode:
                train_index.extend(self.episode2step[i])
        return train_index, test_index


# if __name__ == "__main__":
#     from transformers import AutoTokenizer
#     from tqdm import tqdm

#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#     additional_special_tokens = [SCENE_TOKEN]
#     tokenizer.add_special_tokens(
#         {"additional_special_tokens": additional_special_tokens}
#     )
#     dataset = ExploreDataset("../exploregraph_data", tokenizer, 2048)
#     sampler = DistributedSampler(
#         dataset, num_replicas=1, rank=0, shuffle=True, drop_last=False
#     )
#     dataloader = DataLoader(
#         dataset,
#         batch_size=4,
#         pin_memory=True,
#         num_workers=4,
#         sampler=sampler,
#         collate_fn=dataset.collate_wrapper,
#     )

#     for sample in tqdm(dataloader):
#         print(sample)
#         break

# if __name__ == '__main__':
#     # customize a tokenizer (Not sure how to add special tokens)
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     tokenizer.add_special_tokens(
#         {'additional_special_tokens':[
#             SCENE_TOKEN,
#             # FRONTIER_TOKEN,
#             SELECT_TOKEN
#         ]}
#     )
#     dataset = ExploreDataset('data',tokenizer,1024)

#     # train test split
#     train_index, test_index = dataset.split_index()
#     train_dataset = Subset(dataset,train_index)
#     test_dataset = Subset(dataset,test_index)

#     train_loader = DataLoader(train_dataset, batch_size = 2, shuffle = True, collate_fn = dataset.collate_wrapper)
#     batch = next(iter(train_loader))
#     show_sample(batch)
