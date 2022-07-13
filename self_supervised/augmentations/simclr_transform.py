from self_supervised.augmentations.helper import norm_mean_std, get_color_distortion, GaussianBlur
from torchvision import transforms


class SimCLRTransform:
    """
    Transform defined in SimCLR
    https://arxiv.org/pdf/2002.05709.pdf
    """

    def __init__(self, size):
        normalize = norm_mean_std(size)
        if size > 28:  # ImageNet
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=(size, size)),
                    transforms.RandomHorizontalFlip(),
                    get_color_distortion(s=1.0),
                    # transforms.ToTensor(),
                    normalize  # comment this line out for MNIST datasets
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=(size, size)),
                    transforms.RandomHorizontalFlip(),
                    # get_color_distortion(s=1.0),
                    # transforms.ToTensor(),
                    # normalize
                ]
            )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


