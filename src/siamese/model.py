"""The retrieval encoder shared in architecture, but not weights, by both domains."""

import torch
import torchvision
from torch import nn
from torch.utils.checkpoint import checkpoint

# The ladder unfreezes from the top of the network downwards. Index 0 is the
# replaced embedding layer, trainable from the start; ladder_depth counts how
# many residual stages follow it before the rest stays frozen. The stem
# convolution is last and is never reached at the depths used in the thesis.
_LADDER = ("fc", "layer4", "layer3", "layer2", "layer1", "conv1")


class ImpressionEncoder(nn.Module):
    """ResNet-50 embedding one impression domain into the shared feature space.

    Shoeprints and shoemarks each get their own instance: the architecture is
    shared but the weights are not, so each encoder specialises in the
    appearance of its own domain while the loss aligns only their outputs.
    """

    def __init__(
        self,
        embedding_size=128,
        *,
        pre_trained: bool = False,
        frozen: bool = True,
        ladder_depth: int = 0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        weights = "DEFAULT" if pre_trained else None
        model = torchvision.models.resnet50(weights=weights)

        # Replace the classification layer with the embedding layer
        model.fc = nn.Linear(model.fc.in_features, embedding_size)
        model.apply(self.init_weights)

        if pre_trained and frozen:
            for param in model.parameters():
                param.requires_grad = False

        self.model = model
        self.ladder_depth = ladder_depth
        self.gradient_checkpointing = gradient_checkpointing
        self.unfrozen = 0

        if pre_trained:
            # The randomly initialised embedding layer trains from the start
            self.unfreeze_to(1)

    def unfreeze_next(self):
        """Open the next stage down the ladder, if the depth cap allows it."""
        self.unfreeze_to(self.unfrozen + 1)

    def unfreeze_to(self, stages: int):
        """Make the first `stages` ladder entries trainable (idempotent).

        Capped at ladder_depth + 1 (the embedding layer plus that many residual
        stages); a depth of 0 means every stage unfreezes.
        """
        limit = len(_LADDER) if self.ladder_depth == 0 else self.ladder_depth + 1
        stages = min(stages, limit, len(_LADDER))
        for name in _LADDER[:stages]:
            for param in getattr(self.model, name).parameters():
                param.requires_grad = True
        self.unfrozen = max(self.unfrozen, stages)

    def forward(self, x):
        if not (self.gradient_checkpointing and self.training):
            return self.model(x)
        # Recompute each stage's activations during backward instead of holding
        # them, so the deep ladder fits at batch 96 on a 24 GB card. The
        # recompute runs each stage's BatchNorms a second time per step, so
        # running statistics see doubled updates.
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        for layer in (m.layer1, m.layer2, m.layer3, m.layer4):
            x = checkpoint(layer, x, use_reentrant=False)
        x = torch.flatten(m.avgpool(x), 1)
        return m.fc(x)

    def init_weights(self, m):
        """Xavier-initialise the replaced embedding layer."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
