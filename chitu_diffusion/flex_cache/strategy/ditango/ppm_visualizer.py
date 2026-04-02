from typing import Dict, Tuple


def _decision_to_color(decision_code: int) -> Tuple[int, int, int]:
    if decision_code == 0:
        return (160, 160, 160)  # warmup/cooldown: gray
    if decision_code == 1:
        return (30, 180, 80)  # anchor: green
    if decision_code == 2:
        return (50, 120, 230)  # compute: blue
    return (250, 210, 70)  # reuse: yellow


def save_ditango_decision_ppm(
    records: Dict[Tuple[int, int, int], int],
    max_step: int,
    total_layers: int,
    group_num: int,
    ppm_path: str,
    cell_w: int = 8,
    cell_h: int = 8,
) -> None:
    if max_step < 0 or total_layers <= 0 or group_num <= 0:
        return

    width = (max_step + 1) * cell_w
    rows = total_layers * group_num
    height = rows * cell_h
    rgb = bytearray(width * height * 3)

    for step in range(max_step + 1):
        for layer in range(total_layers):
            for group_id in range(group_num):
                code = records.get((step, layer, group_id), 0)
                color = _decision_to_color(code)
                row = layer * group_num + group_id
                x0 = step * cell_w
                y0 = row * cell_h
                for yy in range(y0, y0 + cell_h):
                    row_off = yy * width
                    for xx in range(x0, x0 + cell_w):
                        idx = (row_off + xx) * 3
                        rgb[idx] = color[0]
                        rgb[idx + 1] = color[1]
                        rgb[idx + 2] = color[2]

    # Horizontal separators: group boundary (thin), layer boundary (thicker).
    for row in range(1, rows):
        y = row * cell_h
        is_layer_boundary = (row % group_num) == 0
        if is_layer_boundary:
            line_color = (30, 30, 30)
            line_thickness = 2
        else:
            line_color = (245, 245, 245)
            line_thickness = 1

        for dy in range(line_thickness):
            yy = y + dy
            if yy >= height:
                continue
            row_off = yy * width
            for xx in range(width):
                idx = (row_off + xx) * 3
                rgb[idx] = line_color[0]
                rgb[idx + 1] = line_color[1]
                rgb[idx + 2] = line_color[2]

    with open(ppm_path, "wb") as f:
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        f.write(header)
        f.write(bytes(rgb))
