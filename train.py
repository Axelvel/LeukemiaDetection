import tensorflow as tf
import matplotlib.pyplot as plt

# Training Loop
def training(model, train_loader, val_loader, learning_rate=0.001, num_epochs=10):
    # Loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Metrics to track loss and accuracy
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    # Lists to store metrics
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    print("Training starting...")
    for epoch in range(num_epochs):
        # Reset the metrics at the start of each epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for inputs, labels in train_loader:
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(inputs, training=True)
                loss = loss_object(labels, predictions)

            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)

            # Backward pass (optimize)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            # Record loss and accuracy
            train_loss(loss)
            train_accuracy(labels, predictions)

        # Validation loop
        for val_inputs, val_labels in val_loader:
            # Forward pass
            val_predictions = model(val_inputs, training=False)
            v_loss = loss_object(val_labels, val_predictions)

            # Record loss and accuracy
            val_loss(v_loss)
            val_accuracy(val_labels, val_predictions)

        # Store metrics
        history['train_loss'].append(train_loss.result().numpy())
        history['train_accuracy'].append(train_accuracy.result().numpy())
        history['val_loss'].append(val_loss.result().numpy())
        history['val_accuracy'].append(val_accuracy.result().numpy())

        # Store metrics
        history['loss'].append(train_loss.result().numpy())
        history['accuracy'].append(train_accuracy.result().numpy())
        
        # Print the progress
        print(f"Epoch {epoch + 1}, "
              f"Loss: {train_loss.result()}, "
              f"Accuracy: {train_accuracy.result() * 100}, "
              f"Validation Loss: {val_loss.result()}, "
              f"Validation Accuracy: {val_accuracy.result() * 100}")
        
    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot training and validation accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history['train_accuracy'], label='Training Accuracy', color='blue', marker='o')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='green', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(2, 1, 2)
    plt.plot(history['train_loss'], label='Training Loss', color='red', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', color='orange', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return history