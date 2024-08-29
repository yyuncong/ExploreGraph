#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from easydict import EasyDict

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import (
    build_vision_projector,
    build_mlp_projector,
)


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
        # self.scene_projector = build_mlp_projector(1024, config.hidden_size)
        self.mm_projector = build_vision_projector(config)
        # self.visual_projector = build_mlp_projector(1024, config.hidden_size)
        # self.tactile_projector = build_mlp_projector(1024, config.hidden_size)
        # self.sound_projector = build_mlp_projector(1024, config.hidden_size)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones([200]))

        # self.output_linear = torch.nn.Linear(4096+4096, 1)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().mm_projector(image_features)
        return image_features

    def _insert_feature(
        self, input_embeds, labels, feature, insert_loc, feature_length
    ):
        # print(input_embeds.shape)
        # print(feature_length, feature.shape)
        for (batch_idx, idx), feat, feat_length in zip(
            insert_loc, feature, feature_length
        ):
            assert len(feat.shape) == 2
            # assert input_embeds.shape[1] - idx >= feat_length
            # print(input_embeds.shape[1] - idx, feat_length)
            # input()
            if len(feat.shape) == 2:
                input_embeds[batch_idx, idx : idx + feat_length] = feat[:feat_length]
                if labels is not None:
                    labels[batch_idx, idx : idx + feat_length] = -100
            else:
                input_embeds[batch_idx, idx] = feat
                if labels is not None:
                    labels[batch_idx, idx] = -100
        return input_embeds, labels

    def _insert_feature(
        self, input_embeds, labels, feature, insert_loc, feature_length
    ):
        batch_idx = 0
        for indices, feat, feat_length in zip(insert_loc, feature, feature_length):
            assert len(feat.shape) == 2
            if len(feat.shape) == 2:
                feat = feat.to(input_embeds.dtype)
                input_embeds[batch_idx, indices[:feat_length]] = feat[:feat_length]
                if labels is not None:
                    labels[batch_idx, indices[:feat_length]] = -100
            else:
                input_embeds[batch_idx, indices[0]] = feat
                if labels is not None:
                    labels[batch_idx, indices[0]] = -100
            batch_idx += 1
        return input_embeds, labels

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, feature_dict
    ):
        if feature_dict is None:
            new_input_embeds = self.get_model().embed_tokens(input_ids)
            new_labels = labels.clone() if labels is not None else None
            return (
                None,
                attention_mask,
                past_key_values,
                new_input_embeds,
                new_labels,
                feature_dict,
            )
        else:
            feature_dict = EasyDict(feature_dict)
            new_input_embeds = self.get_model().embed_tokens(input_ids)
            new_labels = labels.clone() if labels is not None else None
            if past_key_values is None:
                feature_dict.scene_feature_proj = self.get_model().mm_projector(
                    feature_dict.scene_feature
                )
                new_input_embeds, new_labels = self._insert_feature(
                    new_input_embeds,
                    new_labels,
                    feature_dict.scene_feature_proj,
                    feature_dict.scene_insert_loc,
                    feature_dict.scene_length,
                )
            return (
                None,
                attention_mask,
                past_key_values,
                new_input_embeds,
                new_labels,
                feature_dict,
            )
