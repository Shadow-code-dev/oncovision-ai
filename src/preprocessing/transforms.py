import torchvision.transforms as transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),

        # Data Augmentation
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        )
    ])