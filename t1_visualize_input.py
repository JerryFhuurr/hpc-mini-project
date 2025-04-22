import numpy as np
import matplotlib.pyplot as plt

# Load data
domain = np.load("114_domain.npy")
interior = np.load("114_interior.npy")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Domain (5 = load bearing, 25 = inside walls, 0 = interior)
im0 = axes[0].imshow(domain, cmap='coolwarm')
axes[0].set_title("Domain (Initial Conditions)")
plt.colorbar(im0, ax=axes[0])

# Interior mask (1 = inside rooms, 0 = other)
im1 = axes[1].imshow(interior, cmap='gray')
axes[1].set_title("Interior Mask")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.show()
