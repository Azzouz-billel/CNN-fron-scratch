import numpy as np
from pipline import CustomCNN

np.random.seed(42)
sample_image = np.random.randn(3, 4, 4)
target_class = 0

# 2. Instantiate Network
model = CustomCNN()

# 3. Step 1: Forward Pass
logits, loss = model.forward(sample_image, target_class)
print(f"Initial Loss      : {loss:.4f}")
print(f"Predicted Logits  :\n{logits.flatten()}")

# 4. Step 2: Backward Pass (Weight updates applied internally)
model.backward(lr=0.05)

# 5. Step 3: Verify Forward Pass after 1 training step
_, loss_after = model.forward(sample_image, target_class)
print(f"Loss after 1 step : {loss_after:.4f}")