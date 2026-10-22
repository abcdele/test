import torch
from torch import nn
from torch.nn import functional as F
from cfg.defaults import LossConfig
from core.utils import box_iou,aligned_ciou
from core.assignment import BalancedAssignment


def sigmoid_focal_loss(logits,target,alpha=0.25,gamma=2.0):
    logits,target=logits.float(),target.float()
    cross_entropy=F.binary_cross_entropy_with_logits(logits,target,reduction='none')
    probability=logits.sigmoid()
    pt=probability*target+(1-probability)*(1-target)
    weights=alpha*target+(1-alpha)*(1-target)
    return weights*(1-pt).pow(gamma)*cross_entropy


def interaction_consistency_loss(tokens,reference):
    if tokens is None or tokens.shape[1]<2:
        return reference.sum()*0
    return (1-F.cosine_similarity(tokens[:,1:].float(),tokens[:,:-1].float(),dim=-1,eps=1e-8)).mean()


class DetectionCriterion(nn.Module):
    def __init__(self,num_classes,cfg=None):
        super().__init__()
        self.num_classes,self.cfg=num_classes,cfg or LossConfig()
        self.matcher=BalancedAssignment(self.cfg.match_topk)

    def _targets(self,targets,predictions):
        prepared=[]
        ih,iw=predictions['image_hw']
        for target in targets:
            boxes=target['boxes'].to(predictions['boxes'].device).float()
            labels=target['labels'].to(boxes.device).long()
            if boxes.ndim!=2 or boxes.shape[1]!=4 or labels.ndim!=1 or len(boxes)!=len(labels):
                raise ValueError('expected boxes [N,4] and labels [N]')
            if self.cfg.targets_normalized:
                boxes=boxes*boxes.new_tensor((iw,ih,iw,ih))
            if not torch.isfinite(boxes).all() or (boxes[:,2:]<=boxes[:,:2]).any():
                raise ValueError('invalid target boxes')
            if len(labels) and ((labels<0).any() or (labels>=self.num_classes).any()):
                raise ValueError('invalid class ids')
            prepared.append({'boxes':boxes,'labels':labels})
        return prepared

    @torch.no_grad()
    def _resolve(self,predictions,truth,priors,indices):
        anchor_ids,truth_ids,cost=indices
        ignored=torch.zeros(len(predictions),dtype=torch.bool,device=predictions.device)
        if len(truth):
            ignored=box_iou(predictions.detach().float(),truth).amax(1)>self.cfg.ignore_iou
        selected=torch.full((len(predictions),),-1,dtype=torch.long,device=predictions.device)
        if not len(anchor_ids):
            return selected,ignored
        anchor_overlap=box_iou(priors,truth)[anchor_ids,truth_ids]
        acceptable=anchor_overlap>=self.cfg.positive_iou
        ignored[anchor_ids[~acceptable]]=True
        candidates=anchor_ids[acceptable]
        targets=truth_ids[acceptable]
        costs=cost[acceptable]
        if len(candidates):
            order=costs.argsort(stable=True)
            candidates,targets=candidates[order],targets[order]
            rank=torch.arange(len(candidates),device=candidates.device)
            first=torch.full((len(predictions),),len(candidates),dtype=torch.long,device=candidates.device)
            first.scatter_reduce_(0,candidates,rank,reduce='amin',include_self=True)
            unique=torch.nonzero(first<len(candidates)).flatten()
            selected[unique]=targets[first[unique]]
        ignored[selected>=0]=False
        return selected,ignored

    def forward(self,predictions,targets):
        logits=predictions['cls_logits']
        if len(targets)!=len(logits):
            raise ValueError('target batch size differs from predictions')
        prepared=self._targets(targets,predictions)
        assignments=self.matcher(predictions['boxes'],predictions['anchors'],prepared)
        classification=logits.sum()*0
        regression=predictions['boxes'].sum()*0
        foreground=0
        for index,(target,assignment) in enumerate(zip(prepared,assignments)):
            selected,ignored=self._resolve(predictions['boxes'][index],target['boxes'],predictions['priors'],assignment)
            positive=selected>=0
            foreground+=int(positive.sum())
            labels=torch.zeros_like(logits[index])
            labels[positive,target['labels'][selected[positive]]]=1
            classification+=sigmoid_focal_loss(logits[index],labels,self.cfg.focal_alpha,self.cfg.focal_gamma)[~ignored].sum()
            if positive.any():
                regression+=(1-aligned_ciou(predictions['boxes'][index,positive],target['boxes'][selected[positive]])).sum()
        denominator=max(foreground,1)
        classification,regression=classification/denominator,regression/denominator
        consistency=interaction_consistency_loss(predictions.get('interaction_tokens'),logits)
        total=self.cfg.cls_weight*classification+self.cfg.box_weight*regression+self.cfg.consistency_weight*consistency
        return {'total':total,'classification':classification,'box':regression,'consistency':consistency,
            'num_positive':logits.new_tensor(foreground)}
