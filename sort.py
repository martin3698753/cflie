import matplotlib.pyplot as plt
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Input Image
input_image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
ax[0].imshow(input_image, cmap='Blues', vmin=1, vmax=16)
ax[0].set_title('Input Image (4x4)', fontsize=12)
for i in range(4):
    for j in range(4):
        ax[0].text(j, i, input_image[i, j], ha='center', va='center', color='black')

# Filter (Kernel)
filter_kernel = np.array([
    [1, 0],
    [0, -1]
])
ax[1].imshow(filter_kernel, cmap='Greens', vmin=-1, vmax=1)
ax[1].set_title('Filter (2x2)', fontsize=12)
for i in range(2):
    for j in range(2):
        ax[1].text(j, i, filter_kernel[i, j], ha='center', va='center', color='black')

# Remove axes
for a in ax:
    a.set_xticks([])
    a.set_yticks([])

plt.suptitle('Step 1: Input Image and Filter', fontsize=16)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as patches  # Import the patches module
import numpy as np

# Input Image
input_image = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# Filter (Kernel)
filter_kernel = np.array([
    [1, 0],
    [0, -1]
])

# Create figure
fig, ax = plt.subplots(2, 3, figsize=(12, 8))

# Input Image
ax[0, 0].imshow(input_image, cmap='Blues', vmin=1, vmax=16)
ax[0, 0].set_title('Input Image', fontsize=10)
for i in range(4):
    for j in range(4):
        ax[0, 0].text(j, i, input_image[i, j], ha='center', va='center', color='black')

# Highlight the first patch
patch = input_image[:2, :2]
ax[0, 1].imshow(input_image, cmap='Blues', vmin=1, vmax=16)
ax[0, 1].add_patch(patches.Rectangle((-0.5, -0.5), 2, 2, edgecolor='red', fill=False, linewidth=2))
ax[0, 1].set_title('Patch (2x2)', fontsize=10)
for i in range(2):
    for j in range(2):
        ax[0, 1].text(j, i, patch[i, j], ha='center', va='center', color='black')

# Filter
ax[0, 2].imshow(filter_kernel, cmap='Greens', vmin=-1, vmax=1)
ax[0, 2].set_title('Filter', fontsize=10)
for i in range(2):
    for j in range(2):
        ax[0, 2].text(j, i, filter_kernel[i, j], ha='center', va='center', color='black')

# Convolution Operation
output_value = np.sum(patch * filter_kernel)
ax[1, 0].text(0.5, 0.5, f'Convolution:\n{patch} * {filter_kernel}\n= {output_value}',
              ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightyellow', edgecolor='black'))

# Output Feature Map
output_feature_map = np.zeros((3, 3))
output_feature_map[0, 0] = output_value
ax[1, 1].imshow(output_feature_map, cmap='Reds', vmin=0, vmax=16)
ax[1, 1].set_title('Output Feature Map', fontsize=10)
ax[1, 1].text(0, 0, f'{output_value:.1f}', ha='center', va='center', color='black')

# Remove unused subplots
ax[1, 2].axis('off')

# Remove axes
for a in ax.flatten():
    a.set_xticks([])
    a.set_yticks([])

plt.suptitle('Step 2: Convolution Operation', fontsize=16)
plt.tight_layout()
plt.show()
