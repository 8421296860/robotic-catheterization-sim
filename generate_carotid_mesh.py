import math
import random
import os

def generate_cylinder_mesh(radius, length, num_segments, z_offset=0, taper=1.0, wobble=0.0):
    vertices = []
    faces = []
    
    # Bottom ring
    for i in range(num_segments):
        angle = 2.0 * math.pi * i / num_segments
        r = radius * (1.0 + random.uniform(-wobble, wobble))
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        vertices.append((x, y, z_offset))
        
    # Top ring
    top_radius = radius * taper
    for i in range(num_segments):
        angle = 2.0 * math.pi * i / num_segments
        r = top_radius * (1.0 + random.uniform(-wobble, wobble))
        x = r * math.cos(angle)
        y = r * math.sin(angle)
        vertices.append((x, y, z_offset + length))
        
    # Faces connecting the rings
    for i in range(num_segments):
        next_i = (i + 1) % num_segments
        bottom1 = i
        bottom2 = next_i
        top1 = num_segments + i
        top2 = num_segments + next_i
        
        # Triangles
        faces.append((bottom1, bottom2, top1))
        faces.append((bottom2, top2, top1))
        
    return vertices, faces

def merge_meshes(meshes):
    all_vertices = []
    all_faces = []
    offset = 0
    for v, f in meshes:
        all_vertices.extend(v)
        for face in f:
            all_faces.append(tuple(idx + offset for idx in face))
        offset += len(v)
    return all_vertices, all_faces

def rotate_and_translate_mesh(vertices, rx, ry, rz, tx, ty, tz):
    new_v = []
    for x, y, z in vertices:
        # Rotate X
        y1 = y * math.cos(rx) - z * math.sin(rx)
        z1 = y * math.sin(rx) + z * math.cos(rx)
        # Rotate Y
        x2 = x * math.cos(ry) + z1 * math.sin(ry)
        z2 = -x * math.sin(ry) + z1 * math.cos(ry)
        # Rotate Z
        x3 = x2 * math.cos(rz) - y1 * math.sin(rz)
        y3 = x2 * math.sin(rz) + y1 * math.cos(rz)
        
        new_v.append((x3 + tx, y3 + ty, z2 + tz))
    return new_v

# Generate Common Carotid Artery (CCA)
v_cca, f_cca = generate_cylinder_mesh(0.04, 0.15, 16, wobble=0.1)

# Internal Carotid Artery (ICA)
v_ica, f_ica = generate_cylinder_mesh(0.025, 0.12, 16, taper=0.8, wobble=0.15)
v_ica = rotate_and_translate_mesh(v_ica, 0, 0.3, 0.2, 0, 0, 0.15)

# External Carotid Artery (ECA)
v_eca, f_eca = generate_cylinder_mesh(0.02, 0.10, 16, taper=0.7, wobble=0.2)
v_eca = rotate_and_translate_mesh(v_eca, 0, -0.4, -0.1, 0, 0, 0.15)

# The base neck/flesh volume to hold the artery
v_neck, f_neck = generate_cylinder_mesh(0.12, 0.3, 24, z_offset=-0.1, taper=0.9, wobble=0.05)

# Merge all
merged_v, merged_f = merge_meshes([(v_cca, f_cca), (v_ica, f_ica), (v_eca, f_eca), (v_neck, f_neck)])

# Rotate whole thing 90 deg around Y so it lies flat on the table
# and shift it down slightly
merged_v = rotate_and_translate_mesh(merged_v, 0, 1.5708, 0, 0, 0, 0)

# Write to OBJ
out_dir = '/home/admin1/Music/prep/cyint/ros2_ws/src/carotid_robot_description/meshes'
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, 'patient_carotid.obj')

with open(out_file, 'w') as f:
    f.write("# Generated Patient Specific Carotid Mesh\\n")
    for v in merged_v:
        f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\\n")
    for face in merged_f:
        # OBJ faces are 1-indexed
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\\n")

print(f"Saved mesh to {out_file}")
