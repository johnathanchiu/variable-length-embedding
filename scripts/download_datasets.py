import torchvision

torchvision.datasets.CelebA("/fsx/home/johnathan/data", split="all", download=True)
torchvision.datasets.INaturalist(root="/fsx/home/johnathan/data", version='2021_valid', download=True)
torchvision.datasets.CIFAR10(root="/fsx/home/johnathan/data", train=True, download=True)

print("Done!")    