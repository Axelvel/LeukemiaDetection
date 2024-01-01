import tensorflow as tf
import matplotlib.pyplot as plt

# Training Loop
def training(model, train_loader, learning_rate=0.001, num_epochs=10):
    # Loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Metrics to track loss and accuracy
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
   
    # Lists to store metrics
    history = {'loss': [], 'accuracy': []}

    for epoch in range(num_epochs):
        # Reset the metrics at the start of each epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

        for inputs, labels in train_loader:
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(inputs)
                loss = loss_object(labels, predictions)

            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)

            # Backward pass (optimize)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Record loss and accuracy
            train_loss(loss)
            train_accuracy(labels, predictions)

        # Store metrics
        history['loss'].append(train_loss.result().numpy())
        history['accuracy'].append(train_accuracy.result().numpy())

    # Plotting
    plt.figure(figsize=(12, 4))

    # Plot training accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy', color='b', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    # Plot training loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss', color='r', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.show()

    return history