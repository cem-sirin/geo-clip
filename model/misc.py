import torch


def load_gps_data(path) -> torch.Tensor:
    """Load GPS data from a file."""
    gps_data = []
    with open(path, "r") as f:
        next(f)  # skip header
        for line in f:
            gps_data.append([float(x) for x in line.strip().split(",")])
    return torch.Tensor(gps_data)


if __name__ == "__main__":
    gps_data = load_gps_data("model/gps_gallery_100K.csv")
    print(f"Loaded {len(gps_data)} GPS coordinates with shape {gps_data.shape}")
