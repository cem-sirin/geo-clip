import torch
import torch.nn as nn
from model.rff.layers import GaussianEncoding

# Constants
A1 = 1.340264
A2 = -0.081106
A3 = 0.000893
A4 = 0.003796
M = torch.sqrt(torch.tensor(3.0)).item() / 2
SF = 66.50336
f1 = lambda theta: A1 + 3 * A2 * theta**2 + theta**6 * (7 * A3 + 9 * A4 * theta**2)
f2 = lambda theta: A1 + A2 * theta**2 + theta**6 * (A3 + A4 * theta**2)


def equal_earth_projection(L):
    lat = L[:, 0].deg2rad()
    lon = L[:, 1].deg2rad()
    theta = (M * lat.sin()).asin()
    x = lon * theta.cos() / (M * f1(theta))
    y = theta * f2(theta)
    return (torch.stack((x, y), dim=1)) * SF / 180


def equal_earth_inverse_projection(L):
    EPS = 1.0e-9
    NITER = 10

    x = L[:, 0] * 180 / SF
    y = L[:, 1] * 180 / SF

    theta = y.clone()
    mask = torch.ones_like(theta).bool()
    dlat = torch.zeros_like(theta)
    for _ in range(NITER):
        dlat[mask] = (theta[mask] * f2(theta[mask]) - y[mask]) / f1(theta[mask])
        theta[mask] -= dlat[mask]
        mask = dlat.abs() >= EPS

        if mask.sum() == 0:
            break

    lon = M * x * f1(theta) / torch.cos(theta)
    lat = ((theta).sin() / M).asin()
    return torch.stack((lat, lon), dim=1).rad2deg()


class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(
            rff_encoding,
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x


class LocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8], earth_proj=True):
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)
        self.earth_proj = earth_proj

        for i, s in enumerate(self.sigma):
            self.add_module("LocEnc" + str(i), LocationEncoderCapsule(sigma=s))

    def forward(self, location):
        location = location.float()
        if self.earth_proj:
            location = equal_earth_projection(location)

        location_features = self._modules["LocEnc0"](location)

        for i in range(1, self.n):
            location_features += self._modules["LocEnc" + str(i)](location)

        return location_features


class LocationDecoderCapsule(nn.Module):
    def __init__(self):
        super(LocationDecoderCapsule, self).__init__()
        self.tail = nn.Sequential(nn.Linear(512, 1024))
        self.capsule = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        x = self.tail(x)
        x = self.capsule(x)
        return x


class LocationDecoder(nn.Module):
    def __init__(self, sigma=[2**8, 2**4, 2**0]):
        super(LocationDecoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module("LocDec" + str(i), LocationDecoderCapsule())

    def forward(self, location):
        location = location.float()
        coords = torch.zeros(location.shape[0], 4)

        for i in range(self.n):
            coords += self._modules["LocDec" + str(i)](location)

        lat = torch.atan(coords[:, 0] / coords[:, 1]).rad2deg()
        lon = torch.atan(coords[:, 2] / coords[:, 3]).rad2deg()

        return torch.stack((lat, lon), dim=1)


if __name__ == "__main__":
    gps_encoder = LocationEncoder()
    gps_encoder.load_state_dict(torch.load("model/weights/location_encoder_weights.pth"))

    loc_data = torch.Tensor([[40.7128, -74.0060], [34.0522, -118.2437]])  # NYC and LA in lat/lon
    output = gps_encoder(loc_data)
    print("Encoded GPS data shape:", output.shape)
