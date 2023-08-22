import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import functional as F
from torchvision.transforms import (Normalize, Compose, RandomResizedCrop, RandomRotation, 
                                   RandomHorizontalFlip, ColorJitter, RandomAffine, 
                                   GaussianBlur, RandomCrop, RandomErasing, InterpolationMode, 
                                   ToTensor, Resize, CenterCrop)
MEAN = (0.3453, 0.3453, 0.3453)#(0.48145466, 0.4578275, 0.40821073)
STD = (0.2566, 0.2566, 0.2566)#(0.26862954, 0.26130258, 0.27577711)
# [0.3453, 0.3453, 0.3453]) tensor([0.2566, 0.2566, 0.2566]
# ([0.4233, 0.4233, 0.4233]) tensor([0.2114, 0.2114, 0.2114])

class AlbumentationsTransform:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, img):
        img = F.to_tensor(img).numpy() * 255  # Convert to numpy array in range [0, 255]
        img = img.transpose(1, 2, 0)  # Convert from CxHxW to HxWxC
        img = self.transform(image=img)['image']
        img = F.to_pil_image(img.astype('uint8'))  # Convert back to PIL image
        return img


def _convert_to_rgb(image):
    return image.convert('RGB')


def customized_augmentation(image_size):
    normalize = Normalize(mean=MEAN, std=STD)
    # Albumentations transformations
    albu_transforms = A.Compose([
        # A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),  # Elastic deformation
        # A.HistogramMatching(p=0.5, reference_images=None),  # Histogram Equalization (note: you need reference images)
        # Add any other albumentations transforms here
        A.GaussianBlur(p=0.2, blur_limit=(3, 7))
    ])

    # torchvision transforms
    pre_transforms = Compose([
        RandomRotation(degrees=15),
        RandomResizedCrop(
            image_size,
            scale=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC,
        ),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.15, contrast=0.15),
        RandomAffine(degrees=0, translate=(0.1, 0.1)),
        RandomCrop(size=image_size, padding=10),
        # GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
    ])

    post_transforms = Compose([
        _convert_to_rgb,
        ToTensor(),
        normalize,
        RandomErasing(p=0.1, scale=(0.01, 0.05), ratio=(0.3, 3.3), value=0, inplace=False),  # Cutout or Random Erasing
    ])

    train_transform = Compose([
        pre_transforms,
        AlbumentationsTransform(albu_transforms),
        post_transforms
    ])
    """
    The AlbumentationsTransform class acts as a bridge between torchvision and albumentations.
    The HistogramMatching transformation in albumentations requires a set of reference images to match the histogram. You'd need to specify that.
    Always ensure that ToTensor() is one of the last transformations, because it changes the data type and order of dimensions.
    Test the transformation pipeline on a few images to ensure it's working as expected.
    """

    # https://www.kaggle.com/code/raddar/popular-x-ray-image-normalization-techniques
    # https://towardsdatascience.com/deep-learning-in-healthcare-x-ray-imaging-part-5-data-augmentation-and-image-normalization-1ead1c02cfe3