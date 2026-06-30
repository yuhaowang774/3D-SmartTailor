import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def render_mesh_to_image(vertices, faces, out_path, title="SMPL-X Mesh", elev=10, azim=-90):
    verts = vertices.squeeze(0).detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 下采样: 只渲染 1/4 的面以加速 (Wireframe替代Trisurf)
    step = max(1, len(faces_np) // 5000)
    selected = faces_np[::step]
    
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                    triangles=selected, cmap='Spectral_r', alpha=0.9,
                    linewidth=0.02, edgecolor='gray', antialiased=False)
    
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    max_range = max(verts[:, 0].ptp(), verts[:, 1].ptp(), verts[:, 2].ptp()) / 2.0
    mid_x, mid_y, mid_z = verts[:, 0].mean(), verts[:, 1].mean(), verts[:, 2].mean()
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()


def render_front_and_side(vertices, faces, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    render_mesh_to_image(vertices, faces, os.path.join(out_dir, 'mesh_front.png'),
                         title="SMPL-X Front View", elev=0, azim=0)
    render_mesh_to_image(vertices, faces, os.path.join(out_dir, 'mesh_side.png'),
                         title="SMPL-X Side View", elev=0, azim=-90)
