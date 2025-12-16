import torch.nn.functional as F
import torch

def predict(model,img_tensor):
    model.eval()
    with torch.no_grad():
        age_logits, race_logits = model(img_tensor)
        age_probs = F.softmax(age_logits, dim=1)
        race_probs = F.softmax(race_logits, dim=1)

        argmax_age = torch.argmax(age_probs, dim=1).item()
        argmax_race = torch.argmax(race_probs, dim=1).item()

        age_conf = age_probs[0][argmax_age].item()
        race_conf = race_probs[0][argmax_race].item()
    return argmax_age, age_conf, argmax_race, race_conf

