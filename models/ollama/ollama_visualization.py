import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# Define the furniture data
furniture_items = [
    {'furniture': 'dining_table', 'x': -0.82, 'y': 0.37, 'z': 0.13, 'width': 1.5, 'height': 0.8, 'depth': 1.2,
     'angle': 0.0, 'color': 'crimson'},
    {'furniture': 'dining_chair', 'x': -1.59, 'y': 0.4, 'z': -0.6, 'width': 0.5, 'height': 0.8, 'depth': 0.7,
     'angle': 0.0, 'color': 'lightgreen'},
    {'furniture': 'dining_chair', 'x': -0.89, 'y': 0.4, 'z': -0.6, 'width': 0.5, 'height': 0.8, 'depth': 0.7,
     'angle': 0.0, 'color': 'lightgreen'},
    {'furniture': 'dining_chair', 'x': -0.18, 'y': 0.4, 'z': -0.6, 'width': 0.5, 'height': 0.8, 'depth': 0.7,
     'angle': 0.0, 'color': 'lightgreen'},
    {'furniture': 'dining_chair', 'x': -1.58, 'y': 0.4, 'z': 0.85, 'width': 0.5, 'height': 0.8, 'depth': 0.7,
     'angle': -3.14, 'color': 'lightgreen'},
    {'furniture': 'dining_chair', 'x': -0.88, 'y': 0.4, 'z': 0.86, 'width': 0.5, 'height': 0.8, 'depth': 0.7,
     'angle': -3.14, 'color': 'lightgreen'},
    {'furniture': 'pendant_lamp', 'x': -1.2, 'y': 0.9, 'z': 0.13, 'width': 0.5, 'height': 1.5, 'depth': 0.8,
     'angle': 0.0, 'color': 'white'},
    {'furniture': 'console_table', 'x': -1.2, 'y': 0.2, 'z': 0.13, 'width': 1.5, 'height': 0.8, 'depth': 0.7,
     'angle': 0.0, 'color': 'gray'}
]

# Create a new plot
fig, ax = plt.subplots(figsize=(10, 8))


# Function to add rotated rectangle
def add_rotated_rect(ax, x, z, width, depth, angle, color, label=None):
    # Calculate the lower-left corner based on center position
    lower_left_x = x - width / 2
    lower_left_z = z - depth / 2
    # Create a rectangle
    rect = patches.Rectangle((lower_left_x, lower_left_z), width, depth,
                             linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
    # Apply rotation
    t = patches.transforms.Affine2D().rotate_around(x, z, angle) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    if label:
        ax.text(x, z, label, ha='center', va='center', fontsize=8, color='black')


# Iterate through each furniture item and add to plot
for item in furniture_items:
    x = item['x']
    z = item['z']
    width = item['width']
    depth = item['depth']
    angle = item['angle']  # Assuming angle is in radians
    color = item['color']
    furniture = item['furniture']

    # Optionally, label certain furniture types
    label = None
    if furniture == 'dining_table':
        label = 'Table'
    elif furniture == 'dining_chair':
        label = 'Chair'
    elif furniture == 'pendant_lamp':
        label = 'Lamp'
    elif furniture == 'console_table':
        label = 'Console'

    add_rotated_rect(ax, x, z, width, depth, angle, color, label)

# Set plot limits (adjust as needed)
all_x = [item['x'] for item in furniture_items]
all_z = [item['z'] for item in furniture_items]
buffer = 2
ax.set_xlim(min(all_x) - buffer, max(all_x) + buffer)
ax.set_ylim(min(all_z) - buffer, max(all_z) + buffer)

# Set labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Z Position')
ax.set_title('2D Furniture Layout (X-Z Plane)')

# Add grid
ax.grid(True, linestyle='--', alpha=0.5)

# Set aspect ratio to equal for accurate representation
ax.set_aspect('equal')

# Show the plot
plt.show()
