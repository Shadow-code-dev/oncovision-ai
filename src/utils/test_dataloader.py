from src.preprocessing.dataloader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders()

# Get one batch
images, labels = next(iter(train_loader))

print("Image batch shape: ", images.shape)
print("Labels: ", labels)