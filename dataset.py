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


class ExploreDataset(Dataset):
    def __init__(
        self,
        src_path,
        tokenizer,
        max_length,
        scene_token=SCENE_TOKEN,
        # frontier_token = FRONTIER_TOKEN,
        select_token=SELECT_TOKEN,
    ):
        scene_path = "/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory"
        self.scene_dir = os.path.join(scene_path, "scene_feature_dict")
        exploration_path = (
            "/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/3d/explore-eqa-test/"
        )
        self.explore_dir = os.path.join(exploration_path, "exploration_data")
        self.tokenizer = tokenizer
        self.scene_token = scene_token
        self.scene_token_id = self.tokenizer(self.scene_token).input_ids[-1]
        # self.frontier_token = frontier_token
        # self.frontier_token_id = self.tokenizer.convert_tokens_to_ids(self.frontier_token)
        # self.select_token = select_token
        # self.select_token_id = self.tokenizer(select_token).input_ids[-1]
        self.max_length = max_length
        self.data = self.load_data()

    def load_data(self):
        # load scene feature into dict
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

        # load episode data: metadata is managed with self.episodes
        self.episodes = []
        data = []
        for i, episode in enumerate(os.listdir(self.explore_dir)):
            epi_path = os.path.join(self.explore_dir, episode)
            # load metadata
            with open(os.path.join(epi_path, "metadata.json"), "r") as f:
                metadata = json.load(f)
            self.episodes.append(metadata)

            # load step data
            steps_data = []
            for step in range(metadata["episode_length"]):
                with open(os.path.join(epi_path, f"{pad_zero(str(step),4)}.json")) as f:
                    stepdata = json.load(f)
                # link each step to its episode
                stepdata["episode_id"] = i
                # add paths for frontiers
                frontier_features = []
                stepdata["frontier_features"] = {}
                for frontier in stepdata["frontiers"]:
                    # placeholder for loading frontier feature
                    rgb_id = frontier["rgb_id"]
                    frontier_folder = os.path.join(epi_path, "frontier_rgb")
                    # load frontier feature
                    # feature = torch.load(os.path.join(frontier_folder, rgb_id.replace(".png", ".pt")),
                    #                         map_location = 'cpu')
                    feature = os.path.join(
                        frontier_folder, rgb_id.replace(".png", ".pt")
                    )
                    # feature = torch.zeros(1024)
                    stepdata["frontier_features"][rgb_id] = feature
                    # front['rgb_id'] = os.path.join(epi_path,'frontier_rgb',front['rgb_id'])
                # remove frontier info, can be removed in case other features needed
                # del stepdata['frontiers']
                steps_data.append(stepdata)
            data.extend(steps_data)

        # link steps to episodes, which can be used for dataset split
        self.episode2step = defaultdict(list)
        for i in range(len(data)):
            self.episode2step[data[i]["episode_id"]].append(i)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            # load a whole episode and each step within it
            step = self.data[idx]
            episode = self.episodes[step["episode_id"]]
            scene = self.scenes[episode["scene"]]

            text = (
                f"Question: {episode['question']}\n"
                + f"Select the frontier/object that would help finding the answer of the question.\n"
            )

            # replace scene graph in each steps with scene feature
            prediction = np.array(step["prediction"])
            object_features = []
            remove_indices = []
            text += "These are the objects already in our scene graph:\n"
            for i, sid in enumerate(step["scene_graph"]):
                if str(sid) not in scene.keys():
                    remove_indices.append(i)
                else:
                    try:
                        object_feature = torch.load(scene[str(sid)], map_location="cpu")
                        object_features.append(object_feature)
                        # TODO: add object class
                        # text += f"object {i}: <scene> {step['scene_graph'][i]['class']}\n"
                        text += f"object {i} <scene> "
                    except:
                        remove_indices.append(i)
            text += "/\n"

            prediction = np.delete(prediction, remove_indices)
            prediction = torch.tensor(prediction)
            # find the index of 1.0 in prediction
            prediction_index = np.where(prediction == 1.0)[0][0]
            if prediction_index < len(object_features):
                answer = f"object {prediction_index}"
            else:
                answer = f"frontier {prediction_index - len(object_features)}"

            # object_features = [scene[str(sid)] for sid in step["scene_graph"]
            #                     if str(sid) in scene.keys()]
            if len(object_features) == 0:
                # construct zero scene feature if all objects are missed
                object_features = torch.zeros((1, 1024))
            else:
                object_features = torch.stack(object_features, dim=0)

            text += "Below are all the frontiers that we can explore:\n"
            frontier_features = []
            for i, frontier in enumerate(step["frontiers"]):
                frontier_features.append(
                    torch.load(
                        step["frontier_features"][frontier["rgb_id"]],
                        map_location="cpu",
                    )
                )
                text += f"frontier {i} <scene> "
            text += "/\n"
            frontier_features = torch.cat(frontier_features, dim=0)

            text += "Answer: "
            text += answer + self.tokenizer.eos_token
            # test

            scene_feature = torch.cat([object_features, frontier_features], dim=0)

            if len(scene_feature) > 60:
                return self.__getitem__(idx - 1)

            step["scene_feature"] = scene_feature
            # remove scene graph id --- remove this if we need to keep id

            # make sure all things are included
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

            # print(self.tokenizer.decode(input_ids[input_ids != self.tokenizer.pad_token_id]))

            # only pick the index of the first occured token
            scene_insert_loc = (
                (input_ids == self.scene_token_id).nonzero()[:, 1].reshape(-1)
            )
            # frontier_insert_loc = (input_ids == self.frontier_token_id).nonzero()[:1,1].reshape(-1).tolist()

            return EasyDict(
                text=text,
                input_ids=input_ids,
                length=length,
                scene_length=len(scene_feature),
                attention_mask=attention_mask,
                scene_feature=scene_feature,
                scene_insert_loc=scene_insert_loc,
                # frontier_feature = step['frontier_features'],
                # frontier_insert_loc = frontier_insert_loc,
            )
        except:
            # print ("cannot find feature %d"%idx)
            return self.__getitem__(idx - 1)
        # except:
        #     print(step['episode_id'])
        #     print(idx)
        #     print(step['scene_graph'])

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

        return EasyDict(
            input_ids=torch.cat([b.input_ids for b in batch])[..., :max_length],
            attention_mask=torch.cat([b.attention_mask for b in batch])[
                ..., :max_length
            ],
            scene_feature=scene_feature,
            scene_insert_loc=scene_insert_loc.to(torch.long),
            scene_length=torch.tensor([b.scene_length for b in batch]),
            # frontier_feature=frontier_feature,
            # frontier_insert_loc = list(chain.from_iterable([[[batch_idx, x] for x in b.frontier_insert_loc] for batch_idx, b in enumerate(batch)])),
            # scene_length = torch.tensor([b.scene_feature.shape[0] for b in batch]),
            # frontier_length = torch.tensor([b.frontier_feature.shape[0] for b in batch])
            max_scene_length=torch.tensor([b.scene_feature.shape[0] for b in batch]),
        )

    # split the dataset by episode id
    def split_index(self, test_ratio=0.3):
        test_num = int(test_ratio * len(self.episodes))
        test_episode = random.sample(range(len(self.episodes)), test_num)
        # print(test_episode)
        test_episode = [
            i
            for i in range(len(self.episodes))
            if int(self.episodes[i]["scene"].split("-")[0]) > 650
        ]
        # print(test_episode)
        train_index, test_index = [], []
        for i in self.episode2step.keys():
            if i in test_episode:
                test_index.extend(self.episode2step[i])
            else:
                train_index.extend(self.episode2step[i])
        return train_index, test_index


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    additional_special_tokens = [SCENE_TOKEN]
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )
    dataset = ExploreDataset("../exploregraph_data", tokenizer, 2048)
    sampler = DistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=True, drop_last=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        pin_memory=True,
        num_workers=4,
        sampler=sampler,
        collate_fn=dataset.collate_wrapper,
    )

    for sample in tqdm(dataloader):
        print(sample)
        break

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
