"""
Absolute dataset:
Grid point 的绝对位置 (x_i, y_i, z_i) (i = 1, ..., N)

六个极端值定义的 Bounding box:
x_f = max(x_1, ..., x_N), x_b = min(x_1, ..., x_N)
y_l = max(x_1, ..., x_N), y_r = min(x_1, ..., x_N)
z_u = max(x_1, ..., x_N), z_d = min(x_1, ..., x_N)

middle point:
x_c = (x_f + x_b) / 2
y_c = (y_l + y_r) / 2
z_c = (z_u + z_d) / 2

relative depth (molecular-scale-aganostic)
relative_position = (x_i - x_c, y_i - y_c, z_i - z_c)
the r of sphere = max(x_f - x_b, y_r - y_l, z_u - z_d) / 2
relative depth = l2-norm(relative_position) / r

input features:
[delta_x_i, delta_y_i, delta_z_i, r_i, ...]
=>
[delta_x_i + d, delta_y_i + d, delta_z_i + d, r_i, ...]

=>
[d] + [delta_x_i, delta_y_i, delta_z_i, r_i, ...]
"""
