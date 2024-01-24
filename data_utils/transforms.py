# Description: This file contains the transforms used for the dataset.
# Borrowed from [VPT](https://github.com/KMnP/vpt). Thanks to the authors.


import torchvision as tv

def get_transforms(split, size, pretrained_model):
    # if using clip backbones, we adopt clip official normalization.
    if pretrained_model in ["vit-b-22k"]:
        normalize = tv.transforms.Normalize(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )
    else:
        normalize = tv.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    # define the sizes used for resizing and cropping
    if size == 224:
        resize_dim = 256
        crop_dim = 224
    elif size == 384:
        resize_dim = 438
        crop_dim = 384

    elif size == 448:
        resize_dim = 512
        crop_dim = 448
    return (
        tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),
                tv.transforms.RandomCrop(crop_dim),
                tv.transforms.RandomHorizontalFlip(0.5),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
        if split == "train"
        else tv.transforms.Compose(
            [
                tv.transforms.Resize(resize_dim),
                tv.transforms.CenterCrop(crop_dim),
                tv.transforms.ToTensor(),
                normalize,
            ]
        )
    )
