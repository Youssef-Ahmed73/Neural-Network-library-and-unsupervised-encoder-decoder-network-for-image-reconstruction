# network.py
import numpy as np

class NeuralNetwork:
    def __init__(self, layers, loss_function):
        """
        layers: list of layer objects (Dense, ReLU, Sigmoid, ...)
        loss_function: instance of loss class (MSE(), CE(), etc.)
        """
        self.layers = layers
        self.loss_function = loss_function
        self.loss_history = []

    def forward(self, X):
        """
        Pass data forward through all layers.
        X: input batch (batch_size, in_features)
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, dL_dA):
        """
        Backpropagate gradient from loss through all layers.
        dL_dA: gradient of loss wrt final output A[L]
        """
        grad = dL_dA
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train(self, X_train, y_train, optimizer, n_epochs, batch_size=32, verbose=False):
        """
        Full training loop:
        - shuffle data
        - create mini-batches
        - forward pass
        - compute loss and gradient
        - backward pass
        - optimizer step
        """
        m = len(X_train)
        

        for epoch in range(n_epochs):
            # Shuffle dataset (X and y in same order)
            idx = np.random.permutation(m)
            X_shuffled = X_train[idx]
            y_shuffled = y_train[idx]

            # Batch loop
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Zero gradients (optional but safe before forward/backward)
                optimizer.zero_grad(self.layers)

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss (optional return/print)
                loss_value = self.loss_function.loss(y_batch, y_pred)
                self.loss_history.append(loss_value)       

                # Loss gradient (dL/dA_last)
                dL_dA = self.loss_function.grad(y_batch, y_pred)

                # Backward pass (fills layer.dW, layer.db)
                self.backward(dL_dA)

                # Update weights (uses gradients computed by backward)
                optimizer.step(self.layers)

            
            if verbose:
                # compute epoch loss on entire dataset (optional)
                y_all = self.forward(X_train)
                epoch_loss = self.loss_function.loss(y_train, y_all)
                print(f"Epoch {epoch+1}/{n_epochs} - loss: {epoch_loss:.6f}")
            

    def predict(self, X):
        """
        Forward pass only (no training)
        """
        return self.forward(X)
    
    def get_loss_history(self):
        """
        Getter for the loss history (all batch losses)
        """
        return self.loss_history