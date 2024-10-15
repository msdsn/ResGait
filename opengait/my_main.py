from modeling.loss_aggregator import LossAggregator
from data.dataset import Dataset
from data.sampler import TripletSampler
from data.transform import get_transform
from data.collate_fn import CollateFn
#import sgd optimizer
from torch.optim import SGD
import torch.nn.functional as F
import torch
from einops import rearrange
from torchvision.models.resnet import BasicBlock
import copy
from torchvision.models.resnet import BasicBlock, ResNet
import torch.nn as nn
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x
class Resnet9(ResNet):
  def __init__(self):
    self.maxpool_flag = True
    super(Resnet9, self).__init__(BasicBlock, [1, 1, 1, 1])
    self.inplanes = self.inplanes
    self.conv1 = BasicConv2d(1, self.inplanes, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(self.inplanes)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block=BasicBlock, planes=64, blocks=1, stride=1, dilate=False)
    self.layer2 = self._make_layer(BasicBlock, 128, 1, 2,  dilate=False)
    self.layer3 = self._make_layer(BasicBlock, 256, 1, 2,  dilate=False)
    self.layer4 = self._make_layer(BasicBlock, 512, 1, 1,  dilate=False)
  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
    if blocks >= 1:
        layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
    else:
        def layer(x): return x
    return layer

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    return x
class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, s, c_in, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, s, c, h, w = x.size()
        x = self.forward_block(x.reshape(-1, c, h, w), *args, **kwargs)
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()
class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p]
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)
class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        return self.pooling_func(seqs, **options)
class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class SeparateBNNecks(nn.Module):
    """
        Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.build_network()
        self.init_parameters()
        trf_cfg = [
            {'type': 'BaseSilCuttingTransform'},
            {'type': 'RandomRotate', 'prob': 0.3},
            {'type': 'RandomErasing', 'prob': 0.3}
        ]
        self.trainer_trfs = get_transform(trf_cfg)
        data_cnfg = {
            'cache': False,
            'dataset_root': './CASIA-B-pkl',
            'dataset_partition': './datasets/CASIA-B/CASIA-B.json',
        }
        self.train_dataset = Dataset(data_cnfg, training=True)
        self.train_sampler = TripletSampler(self.train_dataset, batch_size=[8, 16], batch_shuffle=True)
        collate_cfg = {
            'sample_type': 'fixed_unordered',
            'frames_num_fixed': 30,
        }
        self.collate_fn = CollateFn(self.train_dataset.label_set, collate_cfg)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=self.train_sampler, collate_fn=self.collate_fn, num_workers=4)

        self.device = torch.distributed.get_rank()
        torch.cuda.set_device(self.device)
        self.to(device=torch.device(
            "cuda", self.device))
        
        loss_cfg = [
            {
                'type': 'TripletLoss',
                'log_prefix': 'triplet',
                'margin': 0.2,
                'loss_term_weight': 1.0,
            },
            {
                'type': 'CrossEntropyLoss',
                'log_prefix': 'softmax',
                'scale': 16,
                'loss_term_weight': 1.0,
            }
        ]
        self.loss_aggregator = LossAggregator(loss_cfg)
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=0.1, momentum=0.9, weight_decay=0.0005)

    def build_network(self):
        self.Backbone = Resnet9()
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(parts_num=16, in_channels=512, out_channels=256)
        self.BNNecks = SeparateBNNecks(parts_num=16, class_num=74, in_channels=256)
        #self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def forward(self, inputs):
        ipts, labs, _, _ = inputs
        outs = self.Backbone(ipts)  # [n, c, s, h, w]

        # # Temporal Pooling, TP
        #outs = self.TP(outs, options={"dim": 2})[0]  # [n, c, h, w]
        outs = torch.max(outs, dim=2)[0]
        # # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(ipts,'n s c h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
    
    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    @ staticmethod
    def run_train(model):
        for inputs in model.train_loader:
            seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
            seqs_batch = torch.tensor(seqs_batch).float().cuda()
            print(f"seqs_batch: {seqs_batch.shape}")
            break


model = Baseline()
model.run_train(model)