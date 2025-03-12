# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen
import torch
import os
from tqdm import tqdm
from utils.fourier import ifft
from glob import glob
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class CreateDataset:
    def __init__(
        self,
        data_path,
    ):
        self.data_path = data_path

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        data_path = self.data_path[idx]

        assert data_path.endswith(".pt") or data_path.endswith(
            ".npy"
        ), "Data should be in .pt or .np format"

        if data_path.endswith(".pt"):
            self.x = torch.load(data_path, weights_only=False).cfloat()
        elif data_path.endswith(".npy"):
            self.x = torch.tensor(np.load(data_path)).cfloat()
            data_path = data_path.replace(".npy", ".pt")

        self.x = ifft(self.x)

        if self.x.ndim == 2:
            self.x = self.x[None, ...]

        return self.x, data_path


# Phase unwrapping

"""https://github.com/blakedewey/phase_unwrap/blob/main/phase_unwrap/cli.py"""
GAUSS_STDEV = 10.0


def unwrap_phase(phase_obj):
    print("Unwrapping phase image.")
    phase_data = torch.tensor(phase_obj)
    if phase_data.max() > torch.pi:
        if phase_data.min() >= 0:
            norm_phase = ((phase_data / phase_data.max()) * 2 * np.pi) - np.pi
        else:
            norm_phase = (phase_data / phase_data.max()) * np.pi
    else:
        norm_phase = phase_data

    dim = norm_phase.shape
    tmp = np.array(
        np.array(range(int(np.floor(-dim[1] / 2)), int(np.floor(dim[1] / 2))))
        / float(dim[1])
    )
    tmp = tmp.reshape((1, dim[1]))
    uu = np.ones((1, dim[0]))
    xx = np.dot(tmp.conj().T, uu).conj().T
    tmp = np.array(
        np.array(range(int(np.floor(-dim[0] / 2)), int(np.floor(dim[0] / 2))))
        / float(dim[0])
    )
    tmp = tmp.reshape((1, dim[0]))
    uu = np.ones((dim[1], 1))
    yy = np.dot(uu, tmp).conj().T
    kk2 = xx**2 + yy**2
    hp1 = gauss_filter(dim[0], GAUSS_STDEV, dim[1], GAUSS_STDEV)

    filter_phase = np.zeros_like(norm_phase)
    with np.errstate(divide="ignore", invalid="ignore"):
        if len(dim) < 3:
            lap_sin = -4.0 * (np.pi**2) * icfft(kk2 * cfft(np.sin(norm_phase)))
            lap_cos = -4.0 * (np.pi**2) * icfft(kk2 * cfft(np.cos(norm_phase)))
            lap_theta = np.cos(norm_phase) * lap_sin - np.sin(norm_phase) * lap_cos
            tmp = np.array(-cfft(lap_theta) / (4.0 * (np.pi**2) * kk2))
            tmp[np.isnan(tmp)] = 1.0
            tmp[np.isinf(tmp)] = 1.0
            kx2 = tmp * (1 - hp1)
            filter_phase = np.real(icfft(kx2))
        else:
            for i in range(dim[2]):
                z_slice = norm_phase[:, :, i]
                lap_sin = -4.0 * (np.pi**2) * icfft(kk2 * cfft(np.sin(z_slice)))
                lap_cos = -4.0 * (np.pi**2) * icfft(kk2 * cfft(np.cos(z_slice)))
                lap_theta = np.cos(z_slice) * lap_sin - np.sin(z_slice) * lap_cos
                tmp = np.array(-cfft(lap_theta) / (4.0 * (np.pi**2) * kk2))
                tmp[np.isnan(tmp)] = 1.0
                tmp[np.isinf(tmp)] = 1.0
                kx2 = tmp * (1 - hp1)
                filter_phase[:, :, i] = np.real(icfft(kx2))

    filter_phase[filter_phase > np.pi] = np.pi
    filter_phase[filter_phase < -np.pi] = -np.pi
    filter_phase *= -1.0

    return filter_phase


def cfft(img_array: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img_array)))


def icfft(freq_array: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(freq_array)))


def gauss_filter(dimx: int, stdevx: float, dimy: int, stdevy: float) -> np.ndarray:
    if dimx % 2 == 0:
        centerx = (dimx / 2.0) + 1
    else:
        centerx = (dimx + 1) / 2.0
    if dimy % 2 == 0:
        centery = (dimy / 2.0) + 1
    else:
        centery = (dimy + 1) / 2.0
    kki = np.array(range(1, dimy + 1)).reshape((1, dimy)) - centery
    kkj = np.array(range(1, dimx + 1)).reshape((1, dimx)) - centerx

    h = gauss(kkj, stdevy).conj().T * gauss(kki, stdevx)
    h /= h.sum()
    h /= h.max()
    return h


def gauss(r: np.ndarray, std0: float) -> np.ndarray:
    return np.exp(-(r**2) / (2 * (std0**2))) / (std0 * np.sqrt(2 * np.pi))


def get_data_list(data_path, save_path):
    data_list = glob(f"{data_path}/*")
    existing_files = set(os.path.basename(f) for f in glob(f"{save_path}/*"))
    data_list = [f for f in data_list if os.path.basename(f) not in existing_files]

    return data_list


def load_model(model_path, device):
    model = torch.load(
        f"{model_path}/model.pth", map_location=device, weights_only=False
    )
    model.to(device)
    model.eval()

    return model


def get_data_loader(data_path, batch_size):
    test_ds = CreateDataset(data_path=data_path)

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True,
    )

    return test_loader


def main(data_path, save_path_genphase, save_path_orig_phase, device):
    data_list = get_data_list(data_path, save_path_genphase)
    model = load_model(model_path, device)

    data_loader = get_data_loader(data_list, 64)

    loop = tqdm(data_loader, desc="Processing batches")

    for i, (x, path) in enumerate(loop):
        x = x.to(device)
        with torch.no_grad():
            if x.shape[1] == 1:
                data_with_phase = model.sample(x.abs())
            else:
                for i in range(x.shape[2]):
                    data_with_phase = model.sample(x[:, i, ...].unsqueeze(1))
                    if torch.isnan(data_with_phase).any():
                        data_with_phase[torch.isnan(data_with_phase)] = 0
                    if i == 0:
                        data_with_phase_stack = data_with_phase.unsqueeze(1)
                    else:
                        data_with_phase_stack = torch.cat(
                            (data_with_phase_stack, data_with_phase.unsqueeze(1)), dim=1
                        )
                data_with_phase = data_with_phase_stack

        for i in range(data_with_phase.shape[0]):
            # unwrap the phase
            data_with_phase_unwrapped = unwrap_phase(
                data_with_phase.angle().squeeze()[i].cpu().numpy()
            )
            x_phase_unwrapped = unwrap_phase(x.angle().squeeze()[i].cpu().numpy())

            plt.imsave(
                Path(save_path_genphase, Path(path[i]).stem + ".png"),
                data_with_phase_unwrapped,
            )
            plt.imsave(
                Path(save_path_orig_phase, Path(path[i]).stem + ".png"),
                x_phase_unwrapped,
            )


if __name__ == "__main__":
    model_path = "/Path/to/model"
    data_path = "/Path/to/data"
    save_path_genphase = "/Path/to/save/gen_phase_unwrapped"
    save_path_orig_phase = "/Path/to/save/orig_phase_unwrapped"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main(data_path, save_path_genphase, save_path_orig_phase, device)
