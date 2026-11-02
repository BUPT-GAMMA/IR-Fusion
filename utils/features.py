import heapq
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


NODE_RE = re.compile(r"_m(?P<layer>\d+)_(?P<x>-?\d+(?:\.\d+)?)_(?P<y>-?\d+(?:\.\d+)?)")
VALUE_SUFFIX = {
    "f": 1e-15,
    "p": 1e-12,
    "n": 1e-9,
    "u": 1e-6,
    "m": 1e-3,
    "k": 1e3,
    "meg": 1e6,
    "g": 1e9,
}


@dataclass
class NodeInfo:
    layer: int
    x: float
    y: float


@dataclass
class SpiceData:
    nodes: Dict[str, NodeInfo]
    resistors: List[Tuple[str, str, float]]
    currents: List[Tuple[str, str, float]]
    voltages: List[Tuple[str, str, float]]


def parse_value(token: str) -> float:
    token = token.strip()
    try:
        return float(token)
    except ValueError:
        pass

    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)([a-zA-Z]+)", token)
    if not match:
        raise ValueError(f"cannot parse SPICE value: {token}")
    value = float(match.group(1))
    suffix = match.group(2).lower()
    return value * VALUE_SUFFIX.get(suffix, 1.0)


def parse_node_name(node: str) -> Optional[NodeInfo]:
    if node.lower() in {"0", "gnd", "vss"}:
        return None
    match = NODE_RE.search(node)
    if not match:
        return None
    return NodeInfo(
        layer=int(match.group("layer")),
        x=float(match.group("x")),
        y=float(match.group("y")),
    )


def parse_spice(spice_path: str) -> SpiceData:
    nodes: Dict[str, NodeInfo] = {}
    resistors: List[Tuple[str, str, float]] = []
    currents: List[Tuple[str, str, float]] = []
    voltages: List[Tuple[str, str, float]] = []

    def add_node(name: str) -> None:
        info = parse_node_name(name)
        if info is not None:
            nodes.setdefault(name, info)

    with open(spice_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("*") or line.startswith("."):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            prefix = parts[0][0].upper()
            if prefix not in {"R", "I", "V"}:
                continue
            node1, node2 = parts[1], parts[2]
            value = parse_value(parts[3])
            add_node(node1)
            add_node(node2)
            if prefix == "R":
                resistors.append((node1, node2, value))
            elif prefix == "I":
                currents.append((node1, node2, value))
            else:
                voltages.append((node1, node2, value))
    return SpiceData(nodes=nodes, resistors=resistors, currents=currents, voltages=voltages)


def bounds_from_nodes(nodes: Dict[str, NodeInfo]) -> Tuple[float, float, float, float]:
    if not nodes:
        return 0.0, 1.0, 0.0, 1.0
    xs = [info.x for info in nodes.values()]
    ys = [info.y for info in nodes.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    return x_min, x_max, y_min, y_max


def node_to_pixel(info: NodeInfo, shape: Tuple[int, int], bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
    height, width = shape
    x_min, x_max, y_min, y_max = bounds
    row = int(math.floor((info.x - x_min) / (x_max - x_min) * (height - 1)))
    col = int(math.floor((info.y - y_min) / (y_max - y_min) * (width - 1)))
    return max(0, min(row, height - 1)), max(0, min(col, width - 1))


def _line_pixels(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    pixels: List[Tuple[int, int]] = []

    if dc >= dr:
        err = dc // 2
        row = r0
        col = c0
        while True:
            pixels.append((row, col))
            if col == c1:
                break
            col += sc
            err -= dr
            if err < 0:
                row += sr
                err += dc
    else:
        err = dr // 2
        row = r0
        col = c0
        while True:
            pixels.append((row, col))
            if row == r1:
                break
            row += sr
            err -= dc
            if err < 0:
                col += sc
                err += dr
    return pixels


def resize_map(array: np.ndarray, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if target_size is None or tuple(array.shape[-2:]) == tuple(target_size):
        return array.astype(np.float32, copy=False)
    squeeze = False
    if array.ndim == 2:
        array = array[None, None, :, :]
        squeeze = True
    elif array.ndim == 3:
        array = array[None, :, :, :]
    else:
        raise ValueError(f"expected 2D or CHW array, got shape={array.shape}")
    tensor = torch.from_numpy(array.astype(np.float32, copy=False))
    resized = F.interpolate(tensor, size=target_size, mode="bilinear", align_corners=False).numpy()
    if squeeze:
        return resized[0, 0]
    return resized[0]


def load_csv_map(path: Path, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return resize_map(data, target_size)


def minmax_normalize(array: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    lo = float(array[finite].min())
    hi = float(array[finite].max())
    if hi - lo <= eps:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - lo) / (hi - lo)).astype(np.float32)


def compute_channel_minmax(feature_paths: Sequence[Path], feature_names: Optional[Sequence[str]] = None, eps: float = 1e-12) -> Dict[str, object]:
    channel_min: Optional[np.ndarray] = None
    channel_max: Optional[np.ndarray] = None
    channel_seen: Optional[np.ndarray] = None
    sample_count = 0

    for feature_path in feature_paths:
        features = np.asarray(np.load(feature_path, allow_pickle=False), dtype=np.float32)
        if features.ndim != 3:
            raise ValueError(f"expected CHW feature array in {feature_path}, got shape={features.shape}")
        channels = int(features.shape[0])
        if channel_min is None:
            channel_min = np.full(channels, np.inf, dtype=np.float64)
            channel_max = np.full(channels, -np.inf, dtype=np.float64)
            channel_seen = np.zeros(channels, dtype=bool)
        elif channels != len(channel_min):
            raise ValueError(f"channel count mismatch in {feature_path}: {channels} != {len(channel_min)}")

        flat = features.reshape(channels, -1)
        finite = np.isfinite(flat)
        for channel in range(channels):
            mask = finite[channel]
            if not mask.any():
                continue
            values = flat[channel, mask]
            channel_min[channel] = min(channel_min[channel], float(values.min()))
            channel_max[channel] = max(channel_max[channel], float(values.max()))
            channel_seen[channel] = True
        sample_count += 1

    if channel_min is None or channel_max is None or channel_seen is None or sample_count == 0:
        raise ValueError("cannot compute channel normalization without training feature files")

    channel_min[~channel_seen] = 0.0
    channel_max[~channel_seen] = 0.0
    names = list(feature_names) if feature_names is not None else [f"channel_{idx}" for idx in range(len(channel_min))]
    if len(names) != len(channel_min):
        raise ValueError(f"feature name count {len(names)} does not match channel count {len(channel_min)}")
    return {
        "mode": "channel-wise-minmax",
        "eps": float(eps),
        "sample_count": int(sample_count),
        "feature_names": names,
        "min": [float(value) for value in channel_min],
        "max": [float(value) for value in channel_max],
    }


def apply_channel_minmax(features: np.ndarray, stats: Dict[str, object]) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError(f"expected CHW feature array, got shape={features.shape}")
    channel_min = np.asarray(stats["min"], dtype=np.float32)
    channel_max = np.asarray(stats["max"], dtype=np.float32)
    if features.shape[0] != channel_min.shape[0]:
        raise ValueError(f"normalization channel count {channel_min.shape[0]} does not match features {features.shape[0]}")
    eps = float(stats.get("eps", 1e-12))
    denom = channel_max - channel_min
    normalized = np.zeros_like(features, dtype=np.float32)
    valid = denom > eps
    if np.any(valid):
        normalized[valid] = (features[valid] - channel_min[valid, None, None]) / denom[valid, None, None]
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32, copy=False)


def normalize_feature_file(feature_path: Path, stats: Dict[str, object]) -> None:
    features = np.asarray(np.load(feature_path, allow_pickle=False), dtype=np.float32)
    np.save(feature_path, apply_channel_minmax(features, stats))


def load_channel_minmax(path: Path) -> Dict[str, object]:
    with open(path, "r") as handle:
        stats = json.load(handle)
    if stats.get("mode") != "channel-wise-minmax":
        raise ValueError(f"unsupported normalization mode in {path}: {stats.get('mode')}")
    return stats


def write_channel_minmax(path: Path, stats: Dict[str, object]) -> None:
    with open(path, "w") as handle:
        json.dump(stats, handle, indent=2)


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _accumulate_node_value(
    image: np.ndarray,
    counts: Optional[np.ndarray],
    node: str,
    value: float,
    nodes: Dict[str, NodeInfo],
    bounds: Tuple[float, float, float, float],
) -> None:
    info = nodes.get(node)
    if info is None:
        return
    row, col = node_to_pixel(info, image.shape, bounds)
    image[row, col] += value
    if counts is not None:
        counts[row, col] += 1.0


def _multi_source_dijkstra(data: SpiceData) -> Dict[str, float]:
    sources = sorted({node for node1, node2, _ in data.voltages for node in (node1, node2) if node in data.nodes})
    if not sources:
        return {}

    graph: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for node1, node2, resistance in data.resistors:
        if node1 in data.nodes and node2 in data.nodes:
            weight = max(float(resistance), 1e-30)
            graph[node1].append((node2, weight))
            graph[node2].append((node1, weight))

    dist = {source: 0.0 for source in sources}
    heap = [(0.0, source) for source in sources]
    heapq.heapify(heap)
    while heap:
        cur_dist, node = heapq.heappop(heap)
        if cur_dist != dist.get(node):
            continue
        for nbr, weight in graph.get(node, ()):
            new_dist = cur_dist + weight
            if new_dist < dist.get(nbr, math.inf):
                dist[nbr] = new_dist
                heapq.heappush(heap, (new_dist, nbr))
    return dist


def load_named_node_values(txt_path: Path, value_column: str) -> Dict[str, float]:
    requested = re.sub(r"[^a-z0-9]+", "_", value_column.strip().lower()).strip("_")
    column_index: Optional[int] = None
    values: Dict[str, float] = {}

    with open(txt_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part for part in re.split(r"[\s,]+", line) if part]
            if not parts:
                continue

            if parse_node_name(parts[0]) is None:
                columns = [re.sub(r"[^a-z0-9]+", "_", part.strip().lower()).strip("_") for part in parts]
                if columns[0] in {"node", "name"}:
                    if requested not in columns:
                        raise ValueError(f"column {value_column!r} is missing from {txt_path}")
                    column_index = columns.index(requested)
                continue

            value_index = column_index if column_index is not None else 1
            if value_index >= len(parts):
                continue
            try:
                values[parts[0]] = float(parts[value_index])
            except ValueError:
                continue

    if not values:
        raise ValueError(f"no node values for column {value_column!r} found in {txt_path}")
    return values


def build_spice_maps(
    spice_path: Path,
    image_shape: Tuple[int, int],
    layers: Sequence[int],
    include_shortest_path: bool = True,
    bottom_current_map: Optional[np.ndarray] = None,
    bottom_current_layer: Optional[int] = None,
    node_voltage_path: Optional[Path] = None,
    node_voltage_column: str = "voltage_V",
) -> Tuple[List[np.ndarray], List[str], Tuple[float, float, float, float]]:
    data = parse_spice(str(spice_path))
    bounds = bounds_from_nodes(data.nodes)
    height, width = image_shape
    selected_layers = [int(layer) for layer in layers]

    layer_current = {layer: np.zeros((height, width), dtype=np.float32) for layer in selected_layers}
    layer_resistance = {layer: np.zeros((height, width), dtype=np.float32) for layer in selected_layers}

    for node1, node2, value in data.resistors:
        info1 = data.nodes.get(node1)
        info2 = data.nodes.get(node2)
        if info1 is None or info2 is None:
            continue

        layer1 = int(info1.layer)
        layer2 = int(info2.layer)
        resistance = float(value)
        if layer1 == layer2:
            if layer1 not in layer_resistance:
                continue
            start = node_to_pixel(info1, (height, width), bounds)
            end = node_to_pixel(info2, (height, width), bounds)
            pixels = _line_pixels(start, end)
            if not pixels:
                continue
            contribution = resistance / float(len(pixels))
            for row, col in pixels:
                layer_resistance[layer1][row, col] += contribution
        else:
            endpoint_layers = []
            if layer1 in layer_resistance:
                endpoint_layers.append((layer1, info1))
            if layer2 in layer_resistance:
                endpoint_layers.append((layer2, info2))
            if not endpoint_layers:
                continue
            contribution = resistance / float(len(endpoint_layers))
            for layer, info in endpoint_layers:
                row, col = node_to_pixel(info, (height, width), bounds)
                layer_resistance[layer][row, col] += contribution

    if node_voltage_path is None:
        raise ValueError("current maps require a node solution file containing voltage_V")
    node_voltages = load_named_node_values(Path(node_voltage_path), node_voltage_column)
    for node1, node2, resistance in data.resistors:
        info1 = data.nodes.get(node1)
        info2 = data.nodes.get(node2)
        if info1 is None or info2 is None or node1 not in node_voltages or node2 not in node_voltages:
            continue
        resistance = abs(float(resistance))
        if resistance <= 0.0:
            continue
        branch_current = abs(node_voltages[node1] - node_voltages[node2]) / resistance
        layer1 = int(info1.layer)
        layer2 = int(info2.layer)
        if layer1 == layer2:
            if layer1 not in layer_current:
                continue
            start = node_to_pixel(info1, (height, width), bounds)
            end = node_to_pixel(info2, (height, width), bounds)
            for row, col in _line_pixels(start, end):
                layer_current[layer1][row, col] += branch_current
        else:
            for layer, info in ((layer1, info1), (layer2, info2)):
                if layer not in layer_current:
                    continue
                row, col = node_to_pixel(info, (height, width), bounds)
                layer_current[layer][row, col] += branch_current

    if bottom_current_map is not None and selected_layers:
        current_layer = int(bottom_current_layer if bottom_current_layer is not None else selected_layers[0])
        if current_layer in layer_current:
            layer_current[current_layer] = resize_map(bottom_current_map, (height, width)).astype(np.float32, copy=False)

    layer_shortest = {layer: np.zeros((height, width), dtype=np.float32) for layer in selected_layers}
    if include_shortest_path:
        distances = _multi_source_dijkstra(data)
        counts = {layer: np.zeros((height, width), dtype=np.float32) for layer in selected_layers}
        for node, distance in distances.items():
            if math.isfinite(distance):
                info = data.nodes.get(node)
                if info is None or int(info.layer) not in layer_shortest:
                    continue
                row, col = node_to_pixel(info, (height, width), bounds)
                layer_shortest[int(info.layer)][row, col] += float(distance)
                counts[int(info.layer)][row, col] += 1.0
        for layer in selected_layers:
            mask = counts[layer] > 0
            layer_shortest[layer][mask] /= counts[layer][mask]

    maps = []
    names = []
    for layer in selected_layers:
        maps.append(layer_current[layer])
        names.append(f"current_m{layer}")
    for layer in selected_layers:
        maps.append(layer_resistance[layer])
        names.append(f"spice_resistance_m{layer}")
    for layer in selected_layers:
        maps.append(layer_shortest[layer])
        names.append(f"spice_shortest_path_resistance_m{layer}")
    return maps, names, bounds


def load_node_solution_txt(
    txt_path: Path,
    image_shape: Tuple[int, int],
    layers: Sequence[int],
    bounds: Optional[Tuple[float, float, float, float]] = None,
    value_column: str = "ir_drop_V",
) -> Tuple[List[np.ndarray], List[str]]:
    maps = {int(layer): np.zeros(image_shape, dtype=np.float32) for layer in layers}
    counts = {int(layer): np.zeros(image_shape, dtype=np.float32) for layer in layers}
    parsed_rows = []
    value_numeric_index = None

    def normalise_column(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")

    with open(txt_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part for part in re.split(r"[\s,]+", line) if part]
            if not parts:
                continue
            first = parts[0].lower()
            if first in {"node", "name", "layer"} and any(not _is_float(part) for part in parts[1:]):
                columns = [normalise_column(part) for part in parts]
                candidates = [normalise_column(value_column), "ir_drop_v", "ir_drop", "voltage_drop", "drop"]
                for candidate in candidates:
                    if candidate in columns:
                        column_index = columns.index(candidate)
                        value_numeric_index = max(0, column_index - 1) if first in {"node", "name"} else column_index
                        break
                continue

            node_info = parse_node_name(parts[0])
            numeric = []
            for token in parts[1 if node_info is not None else 0 :]:
                try:
                    numeric.append(float(token))
                except ValueError:
                    pass
            if node_info is not None and numeric:
                value_index = value_numeric_index if value_numeric_index is not None and value_numeric_index < len(numeric) else -1
                parsed_rows.append((node_info.layer, node_info.x, node_info.y, numeric[value_index]))
            elif len(numeric) >= 4:
                a, b, c, value = numeric[0], numeric[1], numeric[2], numeric[-1]
                if int(round(a)) in maps:
                    parsed_rows.append((int(round(a)), b, c, value))
                else:
                    parsed_rows.append((int(round(c)), a, b, value))

    if not parsed_rows:
        return [maps[int(layer)] for layer in layers], [f"numerical_m{int(layer)}" for layer in layers]

    if bounds is None:
        xs = [row[1] for row in parsed_rows]
        ys = [row[2] for row in parsed_rows]
        bounds = (min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1.0, min(ys), max(ys) if max(ys) > min(ys) else min(ys) + 1.0)

    for layer, x, y, value in parsed_rows:
        layer = int(layer)
        if layer not in maps:
            continue
        row, col = node_to_pixel(NodeInfo(layer, x, y), image_shape, bounds)
        maps[layer][row, col] += float(value)
        counts[layer][row, col] += 1.0

    for layer in maps:
        mask = counts[layer] > 0
        maps[layer][mask] /= counts[layer][mask]
    return [maps[int(layer)] for layer in layers], [f"numerical_m{int(layer)}" for layer in layers]


def zero_numerical_maps(image_shape: Tuple[int, int], layers: Sequence[int]) -> Tuple[List[np.ndarray], List[str]]:
    return (
        [np.zeros(image_shape, dtype=np.float32) for _ in layers],
        [f"numerical_m{int(layer)}" for layer in layers],
    )


def build_feature_stack(
    stem: str,
    raw_dir: str,
    target_size: Optional[Tuple[int, int]],
    layers: Sequence[int],
    node_txt_path: Optional[str] = None,
    node_txt_value_column: str = "ir_drop_V",
    require_label: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    raw_dir_path = Path(raw_dir)
    csv_specs = [
        ("raw_eff_dist_csv", raw_dir_path / f"{stem}_eff_dist.csv"),
        ("raw_pdn_density_csv", raw_dir_path / f"{stem}_pdn_density.csv"),
    ]
    feature_maps: List[np.ndarray] = []
    feature_names: List[str] = []
    for name, path in csv_specs:
        if not path.exists():
            raise FileNotFoundError(path)
        feature_maps.append(load_csv_map(path, target_size))
        feature_names.append(name)

    bottom_current_path = raw_dir_path / f"{stem}_current.csv"
    if not bottom_current_path.exists():
        raise FileNotFoundError(bottom_current_path)
    bottom_current_map = load_csv_map(bottom_current_path, target_size)

    label_path = raw_dir_path / f"{stem}_ir_drop.csv"
    if label_path.exists():
        label = load_csv_map(label_path, target_size)[None, :, :]
        image_shape = tuple(label.shape[-2:])
    elif require_label:
        raise FileNotFoundError(label_path)
    else:
        image_shape = tuple(feature_maps[0].shape[-2:])
        label = np.zeros((1, image_shape[0], image_shape[1]), dtype=np.float32)

    spice_maps, spice_names, bounds = build_spice_maps(
        raw_dir_path / f"{stem}.powerrush.sp",
        image_shape=image_shape,
        layers=layers,
        include_shortest_path=True,
        bottom_current_map=bottom_current_map,
        bottom_current_layer=int(layers[0]) if layers else None,
        node_voltage_path=Path(node_txt_path) if node_txt_path else None,
        node_voltage_column="voltage_V",
    )
    feature_maps.extend(spice_maps)
    feature_names.extend(spice_names)

    if node_txt_path and Path(node_txt_path).exists():
        numerical_maps, numerical_names = load_node_solution_txt(
            Path(node_txt_path),
            image_shape,
            layers,
            bounds=bounds,
            value_column=node_txt_value_column,
        )
    else:
        numerical_maps, numerical_names = zero_numerical_maps(image_shape, layers)
    feature_maps.extend(numerical_maps)
    feature_names.extend(numerical_names)

    features = np.stack(feature_maps, axis=0).astype(np.float32)
    return features, label.astype(np.float32), feature_names


def write_feature_names(path: str, names: Iterable[str]) -> None:
    with open(path, "w") as handle:
        json.dump(list(names), handle, indent=2)
