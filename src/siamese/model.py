"""Siamese model implementation."""

import torch
import torchvision
from torch import nn
from torch.utils.checkpoint import checkpoint


class SharedSiamese(nn.Module):
    """Siamese model with shared weights."""

    def __init__(
        self,
        embedding_size=128,
        *,
        pre_trained: bool = False,
        frozen: bool = True,
        refreeze: bool = False,
        permafrost: int = 0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        if pre_trained:
            model = torchvision.models.resnet50(weights="DEFAULT")
        else:
            model = torchvision.models.resnet50(weights=None)

        # Replace final FC layer with embedding layers
        model.fc = nn.Linear(model.fc.in_features, embedding_size)
        model.apply(self.init_weights)

        # Freeze model
        # TODO tidy this code up
        if pre_trained and frozen:
            for param in model.parameters():
                param.requires_grad = False
        self.permafrost = permafrost
        self.refreeze = refreeze
        self.gradient_checkpointing = gradient_checkpointing

        self.model = model
        self.last_frozen = 1

        if pre_trained:
            self.unfreeze_idx(0)

    def unfreeze_next(self):
        self.unfreeze_idx(self.last_frozen)
        self.last_frozen += 1

    # TODO clean this implementation up as the idea of idx doesn't really make any sense
    def unfreeze_idx(self, idx: int):
        layer_mappings = {
            0: self.model.fc,
            1: self.model.layer4,
            2: self.model.layer3,
            3: self.model.layer2,
            4: self.model.layer1,
            5: self.model.conv1,
        }

        if idx > 5 or (self.permafrost > 0 and idx > self.permafrost):
            return

        for param in layer_mappings[idx].parameters():  # pyright: ignore [reportAttributeAccessIssue]
            param.requires_grad = True

        if self.refreeze and idx >= 1 and idx <= 5:
            for param in layer_mappings[idx - 1].parameters():  # pyright: ignore [reportAttributeAccessIssue]
                param.requires_grad = False

    def unfreeze_to(self, idx: int):
        for i in range(idx):
            self.unfreeze_idx(i)
        # Keep the unfreeze_next() cursor consistent so a resumed run
        # continues the ladder instead of replaying it from layer4
        self.last_frozen = max(self.last_frozen, idx)

    def forward(self, x):
        if not (self.gradient_checkpointing and self.training):
            return self.model(x)
        # Checkpoint each ResNet stage so the deep ladder fits at batch 96 on
        # the 24 GB card: activations are recomputed during backward instead of
        # held. Note the recompute runs each stage's BatchNorms a second time
        # per step, so running stats see doubled updates relative to an
        # uncheckpointed run.
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        for layer in (m.layer1, m.layer2, m.layer3, m.layer4):
            x = checkpoint(layer, x, use_reentrant=False)
        x = torch.flatten(m.avgpool(x), 1)
        return m.fc(x)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
