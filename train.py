import tensorflow as tf

def training(model, train_loader):

    # Training Loop
    for inputs, labels in train_loader:
        outputs = model(inputs)
        print(outputs.shape)
