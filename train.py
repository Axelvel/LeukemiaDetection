import tensorflow as tf

def training(model, train_loader):

    # Training Loop
    for inputs, labels in train_loader:
        print(inputs.shape)
        outputs = model(inputs)