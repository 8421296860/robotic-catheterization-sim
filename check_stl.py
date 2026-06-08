import struct
import sys

def get_stl_bounds(filename):
    with open(filename, 'rb') as f:
        header = f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]
        
        min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
        max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
        
        for _ in range(num_triangles):
            f.read(12) # skip normal
            for _ in range(3):
                x, y, z = struct.unpack('<fff', f.read(12))
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                min_z = min(min_z, z)
                max_z = max(max_z, z)
            f.read(2) # skip attribute byte count
            
    return (min_x, max_x), (min_y, max_y), (min_z, max_z)

bounds = get_stl_bounds(sys.argv[1])
print(f"X: {bounds[0][0]:.2f} to {bounds[0][1]:.2f} (size: {bounds[0][1]-bounds[0][0]:.2f})")
print(f"Y: {bounds[1][0]:.2f} to {bounds[1][1]:.2f} (size: {bounds[1][1]-bounds[1][0]:.2f})")
print(f"Z: {bounds[2][0]:.2f} to {bounds[2][1]:.2f} (size: {bounds[2][1]-bounds[2][0]:.2f})")
