from src.models.model import get_efficientnet, get_resnet

def check_model(model, model_name):
    print(f"\n Checking model: {model_name}")

    total_params = 0
    trainable_params = 0
    trainable_layers = []

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params
            trainable_layers.append(name)

    # Print results
    print(f"\n Total parameters: {total_params:,}")
    print(f"\n Trainable parameters: {trainable_params:,}")

    percentage = (trainable_params / total_params) * 100
    print(f"\n Percentage of trainable parameters: {percentage:.2f}%")

    print("\n Trainable layers:")
    for layer in trainable_layers:
        print("    ", layer)

def main():
    efficientnet = get_efficientnet()
    check_model(efficientnet, "EfficientNet")

    resnet = get_resnet()
    check_model(resnet, "ResNet")

if __name__ == "__main__":
    main()