import torch


def center_size(boxes):
    return torch.cat(((boxes[...,:2]+boxes[...,2:])*0.5,boxes[...,2:]-boxes[...,:2]),-1)


class BalancedAssignment:
    def __init__(self,topk=4):
        if topk<1:
            raise ValueError('topk must be positive')
        self.topk=topk

    @torch.no_grad()
    def __call__(self,predicted_boxes,anchor_boxes,targets):
        matches=[]
        for predictions,target in zip(predicted_boxes,targets):
            truth=target['boxes'].to(predictions.device).float()
            count,objects=len(predictions),len(truth)
            if not objects:
                empty=torch.empty(0,dtype=torch.long,device=predictions.device)
                matches.append((empty,empty, predictions.new_empty(0)))
                continue
            if count==0:
                raise ValueError('cannot assign targets without anchors')
            truth_cs=center_size(truth)
            prediction_cost=torch.cdist(center_size(predictions.detach().float()),truth_cs,p=1)
            anchor_cost=torch.cdist(anchor_boxes.float(),truth_cs,p=1)
            k=min(self.topk,count)
            pred_candidates=prediction_cost.argsort(dim=0,stable=True)[:k]
            anchor_candidates=anchor_cost.argsort(dim=0,stable=True)[:k]
            candidates=torch.cat((pred_candidates,anchor_candidates),0)
            labels=torch.arange(objects,device=predictions.device)[None].expand_as(candidates)
            pairs=(candidates*objects+labels).flatten().unique(sorted=True)
            anchors,objects_idx=pairs//objects,pairs%objects
            quality=prediction_cost[anchors,objects_idx]+anchor_cost[anchors,objects_idx]
            matches.append((anchors,objects_idx,quality))
        return matches
