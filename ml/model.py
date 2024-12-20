"""Structure of the Model"""

from torch import nn


class MovieRecommendation(nn.Module):
    """Define a Model for Movie Recommendations"""

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5 * 1, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
