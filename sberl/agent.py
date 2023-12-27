

from torch.distributions.normal import Normal

from sberl.model import PolicyModel

import torch


class AgentPPO:

    def __init__(self, model: PolicyModel):
        self.model = model

    def choose_action(self, inputs, training=False):
        inputs = torch.tensor(inputs)
        if inputs.ndim < 2:
            inputs = inputs.unsqueeze(0)

        # batch_size = inputs.shape[0]

        means, covar_matrix = self.model.get_policy(inputs)
        normal_distribution = Normal(means, covar_matrix)

        actions = torch.clamp(normal_distribution.sample(), -3., 3.)
        log_probs = normal_distribution.log_prob(actions)

        values = self.model.get_value(inputs)

        if not training:
            return {'actions': actions.cpu().numpy().tolist()[0],
                    'log_probs': log_probs[0].detach().cpu().numpy(),
                    'values': values[0].detach().cpu().numpy()}
        else:
            return {'distribution': normal_distribution, 'values': values}
