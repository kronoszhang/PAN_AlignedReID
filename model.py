import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from resnet import resnet50


class Model(nn.Module):
  def __init__(self, num_classes=None):
    super(Model, self).__init__()
    self.base = resnet50(pretrained=True)
    planes = 2048

    # affine layers
    self.affine_layer1_Conv = nn.Conv2d(512*4, 128, kernel_size=1, stride=1, padding=0, bias=False)
    self.affine_layer1_BN = nn.BatchNorm2d(128)

    self.affine_layer2_Conv = nn.Conv2d(128, 6, kernel_size=1, stride=1, padding=0, bias=False)
    self.affine_layer2_BN = nn.BatchNorm2d(6)
    self.affine_layer2_AvgPool = nn.AdaptiveAvgPool2d((1, 1))

    init.normal_(self.affine_layer1_Conv.weight, std=0.001)
    # init.constant_(self.affine_layer1_Conv.bias, 0)
    init.normal_(self.affine_layer2_Conv.weight, std=0.001)
    # init.constant_(self.affine_layer2_Conv.bias, 0)

    if num_classes is not None:
      self.fc = nn.Linear(planes, num_classes)
      init.normal_(self.fc.weight, std=0.001)
      init.constant_(self.fc.bias, 0)

  def forward(self, x):
    """
    Returns:
      global_feat: shape [N, C]
    """
    # shape [N, C, H, W]
    feat, block2 = self.base(x)
    global_feat = F.avg_pool2d(feat, feat.size()[2:])
    # shape [N, C]
    global_feat = global_feat.view(global_feat.size(0), -1)

    # use `block4 result, i.e. feat` to conduct `block2` affine transformation
    affine = self.affine_layer1_Conv(feat)
    affine = self.affine_layer1_BN(affine)
    affine = self.affine_layer2_Conv(affine)
    affine = self.affine_layer2_BN(affine)
    affine = self.affine_layer2_AvgPool(affine)# shape: N*1*1*6
    affine = affine.float()

    # reshape this to affine para
    bz = affine.shape[0]
    affine = affine.reshape(bz, 2, 3)


    # use this to run affine transform in `block2`
    affine_result = torch.randn((block2.shape), requires_grad=True) # need backword, use this is only to get a torch.variable

    #grid= []
    #output = []
    #for i in range(bz):
      #grid.append(F.affine_grid(affine[i].unsqueeze(0), block2[i].unsqueeze(0).size()))
      #output.append(F.grid_sample(block2[i].unsqueeze(0), grid[i]))
      #affine_result[i] = output[0]
    grid = F.affine_grid(affine, block2.size())
    output = F.grid_sample(block2, grid)
    affine_result = output

    # aligned branch
    affine_result = affine_result.cuda()
    affine_result = self.base.layer3(affine_result)
    affine_result = self.base.layer4(affine_result)

    if hasattr(self, 'fc'):
      logits = self.fc(global_feat)
      affine_logits = self.fc(global_feat)
      return global_feat, affine_result, logits, affine_logits

    return global_feat, affine_result
