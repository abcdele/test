import math
import torch
from torch import nn
from core.utils import classwise_nms


def fuse_actionness_category_logits(category_logits, actionness_logits):
    category, actionness = category_logits.float(), actionness_logits.float()
    return -torch.logaddexp(torch.logaddexp(-category,-actionness),-category-actionness)


class FeatureTower(nn.Sequential):
    def __init__(self,channels,depth,normalization='none',activation='relu'):
        layers=[]
        for _ in range(depth):
            layers.append(nn.Conv2d(channels,channels,3,padding=1,bias=normalization=='none'))
            if normalization=='batch':
                layers.append(nn.BatchNorm2d(channels))
            elif normalization=='group':
                layers.append(nn.GroupNorm(math.gcd(32,channels),channels))
            elif normalization!='none':
                raise ValueError('invalid normalization')
            if activation=='relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation=='silu':
                layers.append(nn.SiLU(inplace=True))
            elif activation=='gelu':
                layers.append(nn.GELU())
            else:
                raise ValueError('invalid activation')
        super().__init__(*layers)
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.normal_(layer.weight,std=0.01)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)


class DecoupledAnchorHead(nn.Module):
    def __init__(self,channels,num_classes,anchors,depth=4,cls_depth=None,reg_depth=None,
                 normalization='none',activation='relu',center_clamp=32.0):
        super().__init__()
        shapes=torch.as_tensor(anchors,dtype=torch.float32)
        if shapes.ndim==1:
            shapes=shapes[:,None].expand(-1,2).clone()
        if shapes.ndim!=2 or shapes.shape[1]!=2 or (shapes<=0).any():
            raise ValueError('anchors must be square sizes or [width,height] pairs')
        self.num_classes,self.num_anchors=num_classes,len(shapes)
        self.center_clamp=center_clamp
        self.register_buffer('anchor_sizes',shapes)
        self.classification=FeatureTower(channels,depth if cls_depth is None else cls_depth,normalization,activation)
        self.regression=FeatureTower(channels,depth if reg_depth is None else reg_depth,normalization,activation)
        self.cls_pred=nn.Conv2d(channels,self.num_anchors*num_classes,3,padding=1)
        self.act_pred=nn.Conv2d(channels,self.num_anchors,3,padding=1)
        self.box_pred=nn.Conv2d(channels,self.num_anchors*4,3,padding=1)
        nn.init.normal_(self.cls_pred.weight,std=0.01)
        nn.init.constant_(self.cls_pred.bias,math.log(0.01/0.99))
        nn.init.normal_(self.box_pred.weight,std=0.01)
        nn.init.zeros_(self.box_pred.bias)

    def make_grid(self,h,w,image_hw,device):
        y,x=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
        grid=torch.stack((x,y),-1).float()[:,:,None].expand(h,w,self.num_anchors,2).reshape(-1,2)
        sizes=self.anchor_sizes[None,None].expand(h,w,-1,-1).reshape(-1,2)
        stride=grid.new_tensor((image_hw[1]/w,image_hw[0]/h))
        return grid,sizes,stride

    def decode(self,anchors,deltas):
        deltas=deltas.float()
        offsets=deltas[...,:2]*anchors[...,2:]
        if self.center_clamp is not None:
            offsets=offsets.clamp(-self.center_clamp,self.center_clamp)
        centers=anchors[...,:2]+offsets
        wh=anchors[...,2:]*deltas[...,2:].clamp(min=-20,max=math.log(1000/16)).exp()
        return torch.cat((centers-wh/2,centers+wh/2),-1)

    def forward(self,x,image_hw):
        b,_,h,w=x.shape
        cls_features=self.classification(x)
        reg_features=self.regression(x)
        def flatten(value,dim):
            return value.reshape(b,self.num_anchors,dim,h,w).permute(0,3,4,1,2).reshape(b,-1,dim)
        categories=flatten(self.cls_pred(cls_features),self.num_classes)
        actionness=flatten(self.act_pred(reg_features),1)
        deltas=flatten(self.box_pred(reg_features),4)
        grid,sizes,stride=self.make_grid(h,w,image_hw,x.device)
        anchors=torch.cat(((grid+0.5)*stride,sizes),-1)
        priors=torch.cat((anchors[:,:2]-sizes/2,anchors[:,:2]+sizes/2),-1)
        return {'cls_logits':fuse_actionness_category_logits(categories,actionness),
            'category_logits':categories,'actionness_logits':actionness,'box_deltas':deltas,
            'boxes':self.decode(anchors[None],deltas),'anchors':anchors,'priors':priors,
            'grid':grid,'sizes':sizes,'stride':stride,'feature_hw':(h,w),'image_hw':tuple(image_hw)}

    @torch.no_grad()
    def postprocess(self,predictions,score_threshold=0.005,nms_threshold=0.5,
                    max_detections=0,pre_nms_topk=1000,normalized=False):
        results=[]
        ih,iw=predictions['image_hw']
        for boxes,logits in zip(predictions['boxes'],predictions['cls_logits']):
            probability=logits.sigmoid().flatten()
            count=min(pre_nms_topk,len(boxes)) if pre_nms_topk>0 else len(probability)
            scores,indices=probability.topk(count,sorted=True)
            valid=scores>score_threshold
            scores,indices=scores[valid],indices[valid]
            labels=indices%self.num_classes
            selected=boxes[indices//self.num_classes].clone()
            selected[:,0::2].clamp_(0,iw)
            selected[:,1::2].clamp_(0,ih)
            valid=(selected[:,2:]>selected[:,:2]).all(-1)
            selected,scores,labels=selected[valid],scores[valid],labels[valid]
            keep=classwise_nms(selected,scores,labels,nms_threshold,max_detections)
            selected=selected[keep]
            if normalized:
                selected=selected/selected.new_tensor((iw,ih,iw,ih))
            results.append({'boxes':selected,'scores':scores[keep],'labels':labels[keep]})
        return results

    def as_anchor_tensor(self,predictions):
        b=predictions['cls_logits'].shape[0]
        h,w=predictions['feature_hw']
        values=torch.cat((predictions['box_deltas'],predictions['actionness_logits'],predictions['category_logits']),-1)
        return values.reshape(b,h,w,self.num_anchors,-1).permute(0,3,4,1,2).reshape(b,-1,h,w)
