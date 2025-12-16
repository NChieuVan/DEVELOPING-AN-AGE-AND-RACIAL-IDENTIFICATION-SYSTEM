from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import transforms
import os

class UTKFaceDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.list_files = os.listdir(img_dir)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        self.transform = transform


    def age_to_group(self, age): 
        """
        Docstring for age_to_group
            0 - 10 tuổi (Trẻ em)
            11 - 19 tuổi (Thanh thiếu niên)
            20 - 30 tuổi (Thanh niên)
            31 - 40 tuổi (Trung niên sớm)
            41 - 50 tuổi (Trung niên)
            50 - 69 tuổi (Người lớn tuổi)
            70+ tuổi (Người già)
       
        """
        if age <= 10:
            return 0
        elif age <= 19:
            return 1
        elif age <= 30:
            return 2
        elif age <= 40:
            return 3
        elif age <= 50:
            return 4
        elif age <= 69:
            return 5
        else:
            return 6
        
    def process_labels(self,filename:str)->tuple:
        """
        - Extract age and race from filename
            - Age groups: 0-6 as defined in age_to_group
            - Race: 0-4 (0: White, 1: Black, 2: Asian, 3: Indian, 4: Others)
            - eg: 25_1_0_20170116174525125.jpg -> age:25, race:0
        - Returns: age_group, race
        """
        for filename in os.listdir(self.img_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                parts = filename.split('_') # ex: 25_1_0_20170116174525125.jpg; return list ['25','1','0','20170116174525125.jpg']
                age = int(parts[0])
                race = int(parts[2])
                age_group = self.age_to_group(age)
        return age_group,race

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, idx):
        file_name = self.list_files[idx]
        img_path = os.path.join(self.img_dir, file_name)
        image = Image.open(img_path).convert("RGB")
        age,race = self.process_labels(file_name)
        if self.transform:
            image = self.transform(image)
        return image, age, race
    
if __name__ == "__main__":
    dataset = UTKFaceDataset(img_dir="data/UTKFace")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
