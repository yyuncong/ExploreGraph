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
        # Jiachen TODO: add your parameter here
        top_k_categories=5,
        num_egocentric_views=5,
        split="train",
    ):
        # scene_path = "/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory"
        self.scene_dir = os.path.join(scene_path, "scene_feature_dict_merged")
        print(self.scene_dir)
        self.ranking_path = os.path.join(scene_path, "selected_candidates.json")
        # exploration_path = (
        #     "/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/3d/explore-eqa-test/"
        # )
        self.obj_bbox_dir = "/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d/hm3d_obj_bbox_merged"
        self.explore_dir = os.path.join(exploration_path, "exploration_data_new")
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

    def load_step(self, step_path):
        with open(step_path, "r") as f:
            stepdata = json.load(f)

        epi_path = "/".join(step_path.split("/")[:-1])
        step_file_name = step_path.split("/")[-1]
        step = int(step_file_name.split(".")[0])

        # add paths for frontiers
        stepdata["frontier_features"] = {}
        frontier_folder = os.path.join(epi_path, "frontier_rgb")
        for frontier in stepdata["frontiers"]:
            # placeholder for loading frontier feature
            rgb_id = frontier["rgb_id"]
            # load frontier feature
            # feature = torch.load(os.path.join(frontier_folder, rgb_id.replace(".png", ".pt")),
            #                         map_location = 'cpu')
            feature = os.path.join(frontier_folder, rgb_id.replace(".png", ".pt"))
            # feature = torch.zeros(1024)
            stepdata["frontier_features"][rgb_id] = feature
            # front['rgb_id'] = os.path.join(epi_path,'frontier_rgb',front['rgb_id'])
        # remove frontier info, can be removed in case other features needed
        # del stepdata['frontiers']
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
        self.scenes = {}
        for scene in os.listdir(self.scene_dir):
            self.scenes[scene] = {}
            scene_fold = os.path.join(self.scene_dir, scene)
            # need to confirm: if object in different scene should have different features
            for object_f in os.listdir(scene_fold):
                object_id = object_f[:-3]
                try:
                    # object_feature  = torch.load(os.path.join(scene_fold, object_f),
                    #                             map_location = 'cpu')
                    # self.scenes[scene][object_id] = object_feature
                    self.scenes[scene][object_id] = os.path.join(scene_fold, object_f)
                except:
                    continue

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

    # def load_data(self):

    #     # Jiachen TODO: load your "question/scene to ranking json" here
    #     with open(self.ranking_path, "r") as f:
    #         self.candidate_rankings = json.load(f)
    #     # load scene feature into dict
    #     self.scenes = {}
    #     for scene in os.listdir(self.scene_dir):
    #         self.scenes[scene] = {}
    #         scene_fold = os.path.join(self.scene_dir, scene)
    #         # need to confirm: if object in different scene should have different features
    #         for object_f in os.listdir(scene_fold):
    #             object_id = object_f[:-3]
    #             try:
    #                 # object_feature  = torch.load(os.path.join(scene_fold, object_f),
    #                 #                             map_location = 'cpu')
    #                 # self.scenes[scene][object_id] = object_feature
    #                 self.scenes[scene][object_id] = os.path.join(scene_fold, object_f)
    #             except:
    #                 continue

    #     self.obj_json_map = {}
    #     for obj_json in os.listdir(self.obj_bbox_dir):
    #         scene_id = obj_json.split(".")[0]
    #         self.obj_json_map[scene_id] = os.path.join(self.obj_bbox_dir, obj_json)

    #     # load episode data: metadata is managed with self.episodes
    #     # TODO later: Remove num skipped to improve error handling
    #     self.episodes = []
    #     data = []
    #     num_skipped = 0
    #     for i, episode in enumerate(os.listdir(self.explore_dir)):
    #         i -= num_skipped
    #         epi_path = os.path.join(self.explore_dir, episode)
    #         # load metadata
    #         try:
    #             with open(os.path.join(epi_path, "metadata.json"), "r") as f:
    #                 metadata = json.load(f)
    #         except:
    #             num_skipped += 1
    #             continue
    #         self.episodes.append(metadata)

    #         # load step data
    #         steps_data = []
    #         for step in range(metadata["episode_length"]):
    #             with open(os.path.join(epi_path, f"{pad_zero(str(step),4)}.json")) as f:
    #                 stepdata = json.load(f)
    #             # link each step to its episode
    #             stepdata["episode_id"] = i
    #             stepdata["target_obj_class"] = metadata["target_obj_class"]

    #             # add paths for frontiers
    #             frontier_features = []
    #             stepdata["frontier_features"] = {}
    #             frontier_folder = os.path.join(epi_path, "frontier_rgb")
    #             for frontier in stepdata["frontiers"]:
    #                 # placeholder for loading frontier feature
    #                 rgb_id = frontier["rgb_id"]
    #                 # load frontier feature
    #                 # feature = torch.load(os.path.join(frontier_folder, rgb_id.replace(".png", ".pt")),
    #                 #                         map_location = 'cpu')
    #                 feature = os.path.join(
    #                     frontier_folder, rgb_id.replace(".png", ".pt")
    #                 )
    #                 # feature = torch.zeros(1024)
    #                 stepdata["frontier_features"][rgb_id] = feature
    #                 # front['rgb_id'] = os.path.join(epi_path,'frontier_rgb',front['rgb_id'])
    #             # remove frontier info, can be removed in case other features needed
    #             # del stepdata['frontiers']
    #             if stepdata["previous_choice"] is not None:
    #                 stepdata["previous_choice"] = os.path.join(
    #                     frontier_folder,
    #                     stepdata["previous_choice"].replace(".png", ".pt"),
    #                 )

    #             stepdata["egocentric_features"] = {}
    #             for view_idx in range(self.num_egocentric_views):
    #                 egocentric_view_folder = os.path.join(epi_path, f"egocentric")
    #                 featrue = os.path.join(
    #                     egocentric_view_folder, f"{i}_view_{view_idx}.pt"
    #                 )
    #                 stepdata["egocentric_features"][view_idx] = featrue
    #             steps_data.append(stepdata)
    #         data.extend(steps_data)

    #     # link steps to episodes, which can be used for dataset split
    #     self.episode2step = defaultdict(list)
    #     for i in range(len(data)):
    #         self.episode2step[data[i]["episode_id"]].append(i)
    #     return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Jiachen TODO: add your feature to get item
        # which might include the following steps
        # 1. load the full list of ranking
        # 2. remove unseen object categories from the full list
        # 3. Take top k object categories as specified by one of the parameter
        # 4. Format the filtering question with the scene graph class names
        # 5. Format the filtering answer with the filtered ranking list
        # 6. Format the selection question with the filtered ranking list (Need further confirmation)
        # 7. Output filter_input_ids/filter_attention_mask/filter_length for the filtering question as well

        # try:
        # load a whole episode and each step within it
        step_path, episode_id = self.data[idx]
        try:
            step = self.load_step(step_path)
        except:
            index = np.random.choice(self.indices)
            return self.__getitem__(index)
        episode = self.episodes[episode_id]
        try:
            scene = self.scenes[episode["scene"]]
        except:
            index = np.random.choice(self.indices)
            return self.__getitem__(index)
        shuffle = self.random_permute and (self.split == "train")
        # Jiachen TODO 1: load ranking
        ranking = self.candidate_rankings[episode["question"] + "_" + episode["scene"]]
        # collections of features from egocentric view/action memory/scene graph/frontiers
        multi_src_features = []

        with open(self.obj_json_map[episode["scene"]]) as f:
            obj_json = json.load(f)
        obj_map = {obj["id"]: obj["class_name"] for obj in obj_json}

        text = f"Question: {episode['question']}\n"

        if self.egocentric_views:
            try:
                egocentric_text, egocentric_features = prepare_egocentric_view(
                    step["egocentric_features"]
                )
            except:
                index = np.random.choice(self.indices)
                return self.__getitem__(index)
            text += egocentric_text
            multi_src_features.append(egocentric_features)

        text += f"Select the frontier/object that would help finding the answer of the question.\n"

        if self.action_memory:
            memory_text, memory_feature = prepare_action_memory(step["previous_choice"])
            text += memory_text
            multi_src_features.append(memory_feature)

        # replace scene graph in each steps with scene feature
        # Jiachen TODO 2: extract seen object categories at the same time
        prediction = np.array(step["prediction"])
        object_features, object_classes, keep_indices = [], [], []
        object_index = 0
        class2object = defaultdict(list)
        for i, sid in enumerate(step["scene_graph"]):
            if str(sid) not in scene.keys() or str(sid) not in obj_map.keys():
                continue
            else:
                try:
                    object_feature = torch.load(scene[str(sid)], map_location="cpu")
                    keep_indices.append(i)
                    object_classes.append(obj_map[str(sid)])
                    object_features.append(object_feature)
                    class2object[obj_map[str(sid)]].append(object_index)
                    object_index += 1
                except:
                    continue
        # print("original indices:", keep_indices)
        # print("seen categories:", object_classes)

        # Data Problem
        if not (
            np.where(prediction[keep_indices] == 1.0)[0].shape[0]
            + np.where(prediction[len(step["scene_graph"]) :] == 1.0)[0].shape[0]
            == 1
        ):
            self.obj_not_found_indices.add(idx)
            index = np.random.choice(self.indices)
            return self.__getitem__(index)

        if self.prefiltering:
            # 1. filter unseen object categories in ranking
            ranking = [cls for cls in ranking if cls in class2object.keys()]
            # print("seen ranking:", ranking)
            # 2. take top k object categories
            ranking = ranking[: self.top_k_categories]
            # print(f"top {self.top_k_categories} ranking:", ranking)
            # 3. reformulate the object indices, classes and features
            keep_indices = [
                keep_indices[obj_idx]
                for cls in ranking
                for obj_idx in class2object[cls]
            ]
            object_classes = [cls for cls in ranking for _ in class2object[cls]]
            object_features = [
                object_features[obj_idx]
                for cls in ranking
                for obj_idx in class2object[cls]
            ]
            # Note that if apply prefiltering, we may have #(objects) < object_index
            # 4. reassign object_index = #(object)
            object_index = len(keep_indices)

            # print("filtered indices:", keep_indices)
            # print("filtered categories:", object_classes)
        # Jiachen TODO: augment data by reindexing objects
        if shuffle:
            # shuffle the index if random_permute is True otherwise keep the original order
            random_object_index = list(range(object_index))
            np.random.shuffle(random_object_index)
            # print(object_index)
            # print('random_object_index', random_object_index)
            # print('indices before shuffle', keep_indices)
            # print('classes before shuffle', object_classes)
            keep_indices = [keep_indices[r_idx] for r_idx in random_object_index]
            object_classes = [object_classes[r_idx] for r_idx in random_object_index]
            object_features = [object_features[r_idx] for r_idx in random_object_index]
            # print('indices after shuffle', keep_indices)
            # print('classes after shuffle', object_classes)

        text += "These are the objects already in our scene graph:\n"
        for i, class_name in enumerate(object_classes):
            text += f"object {i} {class_name} <scene> "

        if object_index == 0:
            text += f"No object available "
            # construct zero scene feature if all objects are missed
            object_features = None
        else:
            object_features = torch.stack(object_features, dim=0)
            # add object features
            multi_src_features.append(object_features)
        text += "/\n"

        """
        try:
            text += "Below are all the frontiers that we can explore:\n"
            if len(step["frontiers"]) > 0:
                frontier_features = []
                for i, frontier in enumerate(step["frontiers"]):
                    frontier_features.append(
                        torch.load(
                            step["frontier_features"][frontier["rgb_id"]],
                            map_location="cpu",
                        )
                    )
                    text += f"frontier {i} <scene> "
                frontier_features = torch.cat(frontier_features, dim=0)
            else:
                text += f"No frontier available "
                frontier_features = None
            text += "/\n"
        except:
            index = np.random.choice(self.indices)
            return self.__getitem__(index)
        """
        # shuffle frontier index
        # print("frontier before shuffle", [frontier['rgb_id'] for frontier in step["frontiers"]])
        frontier_index = list(range(len(step["frontiers"])))
        # shuffle the index if random_permute is True otherwise keep the original order
        if shuffle:
            np.random.shuffle(frontier_index)
        # print("random_frontier_index", frontier_index)
        frontier_text, frontier_features = prepare_frontier(
            step["frontier_features"],
            [step["frontiers"][idx] for idx in frontier_index],
        )
        # print('frontier_text', frontier_text)
        if frontier_text is None:
            index = np.random.choice(self.indices)
            return self.__getitem__(index)
        text += frontier_text
        # add frontier features
        multi_src_features.append(frontier_features)
        # print("prediction before reformat", prediction)
        # prepare prediction and answer
        prediction = np.concatenate(
            (
                prediction[keep_indices],
                prediction[[idx + len(step["scene_graph"]) for idx in frontier_index]],
            )
        )
        # print("reformatted prediction", prediction)
        prediction = torch.tensor(prediction)
        # assert prediction.shape[0] == len(object_features) + len(step["frontiers"])
        assert prediction.shape[0] == object_index + len(step["frontiers"])
        # GPT problem: prefiltering filter out the answer object
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
        """
        if object_features is not None and frontier_features is not None:
            scene_feature = torch.cat([object_features, frontier_features], dim=0)
        elif object_features is not None:
            scene_feature = object_features
        else:
            scene_feature = frontier_features
        
        # there is a bug if we use egocentric views and action memory at the same time
        if self.egocentric_views:
            scene_feature = torch.cat([egocentric_features, scene_feature], dim=0)

        if self.action_memory and memory_feature is not None:
            scene_feature = torch.cat([memory_feature, scene_feature], dim=0)
        """
        # default order: egocentric views -> action memory -> objects -> frontiers
        multi_src_features = [f for f in multi_src_features if f is not None]
        scene_feature = torch.cat(multi_src_features, dim=0)

        if len(scene_feature) > 50:
            # take a random integer index
            # random_idx = np.random.randint(0, len(self.data))
            self.too_many_objects_indices.add(idx)
            index = np.random.choice(self.indices)
            return self.__getitem__(index)
            # return self.__getitem__(random_idx)

        step["scene_feature"] = scene_feature
        # remove scene graph id --- remove this if we need to keep id

        # make sure all things are included
        # print("selection prompt", len(text))
        # print(text)
        if self.max_length <= len(text):
            # print(text)
            self.too_many_objects_indices.add(idx)
            index = np.random.choice(self.indices)
            return self.__getitem__(index)

        assert self.max_length > len(text)
        assert self.max_length > len(
            scene_feature
        )  # make sure that scene feature is never truncated

        text = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = text["input_ids"]
        length = torch.nonzero(input_ids).shape[0]

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
            classes = list(class2object.keys())
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
        # max_frontier_length = max(b.frontier_feature.shape[0] for b in batch)

        scene_feature = torch.zeros((len(batch), max_scene_length, 1024))
        scene_insert_loc = torch.zeros((len(batch), max_scene_length))

        for j, b in enumerate(batch):
            scene_feature[j, : b.scene_feature.shape[0]] = b.scene_feature
            # frontier_feature[j, :b.frontier_feature.shape[0]] = b.frontier_feature
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
                # Jiachen TODO 7
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

    # split the dataset by episode id
    def split_index(self, test_ratio=0.3):
        test_num = int(test_ratio * len(self.episodes))
        test_episode = random.sample(range(len(self.episodes)), test_num)
        test_episode = [
            i
            for i in range(len(self.episodes))
            if int(self.episodes[i]["scene"].split("-")[0]) > 700
        ]
        # print("test episode", test_episode)
        train_index, test_index = [], []
        # print(self.episode2step)
        for i in self.episode2step.keys():
            if i in test_episode:
                test_index.extend(self.episode2step[i])
            else:
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
