import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
from tqdm import tqdm

N_FFT = 128
HOP_LENGTH = N_FFT // 4
BATCH_SIZE = 8
HIDDEN_DIM = 64
LR = 0.01
WINDOW_FN = torch.hann_window

N_BINS = N_FFT // 2 + 1
WINDOW = WINDOW_FN(N_FFT)


def log_transform(waveform):
    return torch.log1p(waveform)


def inv_log_transform(log_waveform):
    return torch.expm1(log_waveform)


def stft_transform(waveform):
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW.to(waveform.device), return_complex=True)
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    mag = log_transform(mag)
    return torch.stack([mag, phase], dim=-1)


def istft_transform(stft):
    mag = stft[..., 0]
    phase = stft[..., 1]
    mag = inv_log_transform(mag)
    stft = torch.polar(mag, phase)
    stft = stft.squeeze(1)
    waveform = torch.istft(stft, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW.to(stft.device))
    return waveform


class AudioDataset(Dataset):
    def __init__(self, x_dir, y_dir, transform=None):
        self.x_dir = x_dir
        self.y_dir = y_dir

        x_files = os.listdir(x_dir)
        y_files = os.listdir(y_dir)

        assert len(x_files) == len(y_files), f"Number of input files ({len(x_files)}) and output files ({len(y_files)}) must be equal"
        self.x_files = x_files
        self.y_files = y_files
        self.transform = transform

    def __len__(self):
        return len(self.x_files)

    def __getitem__(self, idx):
        x_path = os.path.join(self.x_dir, self.x_files[idx])
        y_path = os.path.join(self.y_dir, self.y_files[idx])
        x_waveform, _ = torchaudio.load(x_path)
        y_waveform, _ = torchaudio.load(y_path)
        x_waveform = x_waveform.mean(dim=0, keepdim=True)
        y_waveform = y_waveform.mean(dim=0, keepdim=True)

        x = stft_transform(x_waveform)
        y = stft_transform(y_waveform)

        noise = log_transform(F.relu(inv_log_transform(x[..., 0]) - inv_log_transform(y[..., 0])))
        noise = torch.stack([noise, x[..., 1]], dim=-1)

        x = x.squeeze(0)
        y = y.squeeze(0)
        noise = noise.squeeze(0)

        return x, y, noise


class MifiNoisePredictor(nn.Module):
    def __init__(self):
        super(MifiNoisePredictor, self).__init__()

        self.fc1 = nn.Linear(N_BINS, 128)

        self.conv1 = nn.Conv1d(128, 512, 5, stride=1, padding=4)
        self.conv2 = nn.Conv1d(512, 512, 3, stride=1, padding=2)

        self.gru1 = nn.GRU(512, 512, 1, batch_first=True)
        self.gru2 = nn.GRU(512, 512, 1, batch_first=True)
        self.gru3 = nn.GRU(512, 512, 1, batch_first=True)

        self.fc2 = nn.Linear(512 * 3, N_BINS)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc1(x))
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = x[..., :-4]
        x = torch.tanh(self.conv2(x))
        x = x[..., :-2]
        x = x.permute(0, 2, 1)
        gru1_out, _ = self.gru1(x)
        gru2_out, _ = self.gru2(gru1_out)
        gru3_out, _ = self.gru3(gru2_out)
        x = torch.cat([gru1_out, gru2_out, gru3_out], dim=-1)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        losses = []
        progbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x, y, noise in progbar:
            x = x.to(device)[..., 0]
            y = y.to(device)[..., 0]
            noise = noise.to(device)[..., 0]
            optimizer.zero_grad()
            pred_noise = model(x)
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            progbar.set_postfix({"Loss": current_loss})
            losses.append(loss.item())
        loss = torch.tensor(losses).mean()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

        model.eval()
        losses = []
        with torch.no_grad():
            for x, y, noise in test_dataloader:
                x = x.to(device)[..., 0]
                y = y.to(device)[..., 0]
                noise = noise.to(device)[..., 0]
                pred_noise = model(x)
                loss = criterion(pred_noise, noise)
                losses.append(loss.item())
            loss = torch.tensor(losses).mean()
            print(f"Validation Loss: {loss.item()}")

            if not os.path.exists("models"):
                os.makedirs("models")
            torch.save(model.state_dict(), f"models/model_{epoch+1}.pt")
            print(f"Model saved as model_{epoch+1}.pt")

            if not os.path.exists("results"):
                os.makedirs("results")

            x, y, noise = next(iter(test_dataloader))
            x = x.to(device)
            y = y.to(device)
            noise = noise.to(device)

            x_waveform = istft_transform(x)
            y_waveform = istft_transform(y)

            pred_noise = model(x[..., 0])
            pred_noise_waveform = istft_transform(torch.stack([pred_noise, x[..., 1]], dim=-1))

            denoised = log_transform(F.relu(inv_log_transform(x[..., 0]) - inv_log_transform(pred_noise)))
            denoised = torch.stack([denoised, x[..., 1]], dim=-1)
            denoised_waveform = istft_transform(denoised)

            noise = istft_transform(noise)

            try:
                for i in range(BATCH_SIZE):
                    torchaudio.save(f"results/epoch_{epoch+1}_sample_{i}_input.wav", x_waveform.cpu()[i : i + 1], 16000)
                    torchaudio.save(f"results/epoch_{epoch+1}_sample_{i}_target.wav", y_waveform.cpu()[i : i + 1], 16000)
                    torchaudio.save(f"results/epoch_{epoch+1}_sample_{i}_output.wav", pred_noise_waveform.cpu()[i : i + 1], 16000)
                    torchaudio.save(f"results/epoch_{epoch+1}_sample_{i}_denoised.wav", denoised_waveform.cpu()[i : i + 1], 16000)
                    torchaudio.save(f"results/epoch_{epoch+1}_sample_{i}_noise.wav", noise.cpu()[i : i + 1], 16000)

            except Exception as e:
                print(f"Error saving samples: {e}")
            print(f"Sample results saved for epoch {epoch+1}")


def collate_fn(batch):
    x, y, noise = zip(*batch)
    max_length = max([x_.shape[-2] for x_ in x])
    xs = []
    ys = []
    noises = []
    for x_, y_, noise_ in zip(x, y, noise):
        x_ = F.pad(x_, (0, 0, max_length - x_.shape[-2], 0))
        y_ = F.pad(y_, (0, 0, max_length - y_.shape[-2], 0))
        noise_ = F.pad(noise_, (0, 0, max_length - noise_.shape[-2], 0))
        xs.append(x_)
        ys.append(y_)
        noises.append(noise_)
    x = torch.stack(xs)
    y = torch.stack(ys)
    noise = torch.stack(noises)
    return x, y, noise


if __name__ == "__main__":
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "data")
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    y_train_dir = os.path.join(data_dir, "y_train")
    y_test_dir = os.path.join(data_dir, "y_test")
    train_dataset = AudioDataset(train_dir, y_train_dir)
    test_dataset = AudioDataset(test_dir, y_test_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MifiNoisePredictor().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=100)
