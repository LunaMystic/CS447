# Attack # 
import torch
class FGM():
    def __init__(self, model):
        self.model               = model
        self.backup              = {}
        self.emb_name            = model.embed

    def attack(self, epsilon = 1.,emb_name = 'embed.weight'):

        for name, param in self.model.named_parameters():
            # print(name) -> embed.weight
            # print(param) -> tensor
            # print(param.requires_grad) -> True
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm              = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at          = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self,emb_name = 'embed.weight'):

        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data        = self.backup[name]
        self.backup               = {}