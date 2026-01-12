import os, random
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import hflip, rotate, crop
from torchvision.transforms import ToTensor, RandomCrop, Resize


class TrainDataset(data.Dataset):
    def __init__(self, hazy_path, GT_path):
        super(TrainDataset, self).__init__()
        self.hazy_path = hazy_path
        self.GT_path = GT_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.GT_image_list = os.listdir(GT_path)

        self.is_ohaze='ohaze'in hazy_path.lower() or 'o-haze' in hazy_path.lower()

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        hazy_basename, hazy_ext = os.path.splitext(hazy_image_name)
        split_chars = ['_', '-', '.']
        for char in split_chars:
            if char in hazy_basename:
                parts = hazy_basename.split(char)
                gt_core = [p for p in parts if not p.lower() in ['haze', 'fog', 'hazy', 'foggy']]
                if gt_core:
                    gt_basename = char.join(gt_core)
                else:
                    gt_basename = char.join(parts[1:]) if len(parts) > 1 else hazy_basename
                break
        else:
            gt_basename = hazy_basename


        GT_image_name = f"{gt_basename}{hazy_ext}"

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        GT_image_path = os.path.join(self.GT_path, GT_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        GT = Image.open(GT_image_path).convert('RGB')

        crop_params = RandomCrop.get_params(hazy, [256, 256])
        rotate_params = random.randint(0, 3) * 90

        hazy = crop(hazy, *crop_params)
        GT = crop(GT, *crop_params)

        hazy = rotate(hazy, rotate_params)
        GT = rotate(GT, rotate_params)

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        GT = to_tensor(GT)

        return hazy, GT

    def __len__(self):
        return len(self.hazy_image_list)


class TestDataset(data.Dataset):
    def __init__(self, hazy_path, GT_path):
        super(TestDataset, self).__init__()
        self.hazy_path = hazy_path
        self.GT_path = GT_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.GT_image_list = os.listdir(GT_path)
        self.hazy_image_list.sort()
        self.GT_image_list.sort()

        self.is_ohaze = 'ohaze' in hazy_path.lower() or 'o-haze' in hazy_path.lower()

    def __getitem__(self, index):


        hazy_image_name = self.hazy_image_list[index]


        hazy_basename, hazy_ext = os.path.splitext(hazy_image_name)


        split_chars = ['_', '-', '.']
        for char in split_chars:
            if char in hazy_basename:

                parts = hazy_basename.split(char)

                gt_core = [p for p in parts if not p.lower() in ['haze', 'fog', 'hazy', 'foggy']]
                if gt_core:
                    gt_basename = char.join(gt_core)
                else:

                    gt_basename = char.join(parts[1:]) if len(parts) > 1 else hazy_basename
                break
        else:

            gt_basename = hazy_basename


        GT_image_name = f"{gt_basename}{hazy_ext}"

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        GT_image_path = os.path.join(self.GT_path, GT_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        GT = Image.open(GT_image_path).convert('RGB')

        if self.is_ohaze:
            hazy=hazy.crop((100,100,1950,2048))
            GT=GT.crop((100,100,1950,2048))

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        GT = to_tensor(GT)

        return hazy, GT, hazy_image_name

    def __len__(self):
        return len(self.hazy_image_list)


class ValDataset(data.Dataset):
    def __init__(self, hazy_path, GT_path):
        super(ValDataset, self).__init__()
        self.hazy_path = hazy_path
        self.GT_path = GT_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.GT_image_list = os.listdir(GT_path)
        self.hazy_image_list.sort()
        self.GT_image_list.sort()

        self.is_ohaze = 'ohaze' in hazy_path.lower() or 'o-haze' in hazy_path.lower()

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]


        hazy_basename, hazy_ext = os.path.splitext(hazy_image_name)


        split_chars = ['_', '-', '.']
        for char in split_chars:
            if char in hazy_basename:

                parts = hazy_basename.split(char)

                gt_core = [p for p in parts if not p.lower() in ['haze', 'fog', 'hazy', 'foggy']]
                if gt_core:
                    gt_basename = char.join(gt_core)
                else:

                    gt_basename = char.join(parts[1:]) if len(parts) > 1 else hazy_basename
                break
        else:

            gt_basename = hazy_basename


        GT_image_name = f"{gt_basename}{hazy_ext}"

        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        GT_image_path = os.path.join(self.GT_path, GT_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        GT = Image.open(GT_image_path).convert('RGB')

        if self.is_ohaze:
            hazy=hazy.crop((100,100,1950,2048))
            GT=GT.crop((100,100,1950,2048))

        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        GT = to_tensor(GT)

        return {'hazy': hazy, 'GT': GT, 'filename': hazy_image_name}

    def __len__(self):
        return len(self.hazy_image_list)