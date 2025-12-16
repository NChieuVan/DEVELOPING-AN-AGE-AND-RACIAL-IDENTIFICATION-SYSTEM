import torch.nn.functional as F
import torch

# compute combined cross-entropy loss for age and race
def cross_entropy_loss(logits_age, targets_age,logits_race, targets_race):
    loss_age = F.cross_entropy(logits_age, targets_age) 
    loss_race = F.cross_entropy(logits_race, targets_race)
    return loss_age + loss_race

# compute combined focal loss for age and race
def focal_loss(logits_age, targets_age, logits_race, targets_race, alpha=1, gamma=2, weight_age=None, weight_race=None):
    """
    Focal loss cho bài toán đa nhãn, hỗ trợ class weights cho mất cân bằng:
    - weight_age: tensor (num_age_classes,) hoặc None
    - weight_race: tensor (num_race_classes,) hoặc None
    """
    # Focal loss for age
    ce_age = F.cross_entropy(logits_age, targets_age, reduction='none', weight=weight_age)
    pt_age = torch.exp(-ce_age)
    focal_age = alpha * (1 - pt_age) ** gamma * ce_age
    loss_age = focal_age.mean()

    # Focal loss for race
    ce_race = F.cross_entropy(logits_race, targets_race, reduction='none', weight=weight_race)
    pt_race = torch.exp(-ce_race)
    focal_race = alpha * (1 - pt_race) ** gamma * ce_race
    loss_race = focal_race.mean()

    return loss_age + loss_race