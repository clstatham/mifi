import torch
import torchaudio
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from datetime import datetime
import os
from tqdm import tqdm

N_FFT = 512
HOP_LENGTH = N_FFT // 4
BATCH_SIZE = 16
HIDDEN_SIZE = 512
LR = 0.005
WINDOW_FN = torch.hann_window

N_BINS = N_FFT // 2 + 1
WINDOW = WINDOW_FN(N_FFT)


def stft_transform(waveform):
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW.to(waveform.device), return_complex=True)
    mag = torch.abs(stft)
    phase = torch.angle(stft)
    return torch.stack([mag, phase], dim=-1)


def istft_transform(stft):
    mag = stft[..., 0]
    phase = stft[..., 1]
    stft = torch.polar(mag, phase)
    stft = stft.squeeze(1)
    waveform = torch.istft(stft, n_fft=N_FFT, hop_length=HOP_LENGTH, window=WINDOW.to(stft.device))
    return waveform


def save_ratio_image(tensor, filename):
    tensor = tensor.squeeze(0).cpu()
    tensor = torchvision.transforms.ToPILImage()(tensor)
    tensor.save(filename)


class AudioDataset(Dataset):
    def __init__(self, x_dir, y_dir, transform=None):
        self.x_dir = x_dir
        self.y_dir = y_dir

        x_files = os.listdir(x_dir)
        y_files = os.listdir(y_dir)

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

        noise = F.relu(x[..., 0] - y[..., 0])
        noise_ratio = torch.where(x[..., 0] > 0, noise / x[..., 0], torch.zeros_like(noise))
        noise = torch.stack([noise, x[..., 1]], dim=-1)

        x = x.squeeze(0)
        y = y.squeeze(0)
        noise = noise.squeeze(0)
        noise_ratio = noise_ratio.squeeze(0)

        return x, y, noise, noise_ratio


class MifiNoisePredictor(nn.Module):
    def __init__(self):
        super(MifiNoisePredictor, self).__init__()

        self.fc1 = nn.Linear(N_BINS, HIDDEN_SIZE)
        self.hidden = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE, num_layers=2, batch_first=True, bidirectional=False)
        self.fc2 = nn.Linear(HIDDEN_SIZE, N_BINS)

    def forward(self, x):
        with torch.no_grad():
            x = torch.log1p(x)
            x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True)  # normalize to mean 0, std 1
        x = x.permute(0, 2, 1)
        x = F.relu(self.fc1(x))
        x, _ = self.hidden(x)
        x = F.sigmoid(self.fc2(x))
        x = x.permute(0, 2, 1)
        return x


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join("training", timestamp)
    if not os.path.exists(root):
        os.makedirs(root)
    log_file = open(os.path.join(root, "log.txt"), "w")

    log_file.write(f"Training run {timestamp}\n")
    log_file.write(f"Batch size: {BATCH_SIZE}\n")
    log_file.write(f"Learning rate: {LR}\n")
    log_file.write(f"Hidden size: {HIDDEN_SIZE}\n")
    log_file.write(f"STFT parameters: N_FFT={N_FFT}, HOP_LENGTH={HOP_LENGTH}\n")
    log_file.write("-" * 80 + "\n")

    for epoch in range(num_epochs):
        model.train()
        losses = []
        progbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for x, y, noise, noise_ratio in progbar:
            x = x.to(device)[..., 0]
            y = y.to(device)[..., 0]
            noise = noise.to(device)[..., 0]
            noise_ratio = noise_ratio.to(device)
            optimizer.zero_grad()
            pred_noise = model(x)
            loss = criterion(pred_noise, noise_ratio)
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            progbar.set_postfix({"Loss": current_loss})
            losses.append(loss.item())
        loss = torch.tensor(losses).mean()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
        log_file.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}\n")

        model.eval()
        losses = []
        with torch.no_grad():
            for x, y, noise, noise_ratio in test_dataloader:
                x = x.to(device)[..., 0]
                y = y.to(device)[..., 0]
                noise = noise.to(device)[..., 0]
                noise_ratio = noise_ratio.to(device)
                pred_noise = model(x)
                loss = criterion(pred_noise, noise_ratio)
                losses.append(loss.item())
            loss = torch.tensor(losses).mean()
            print(f"Validation Loss: {loss.item()}")
            log_file.write(f"Validation Loss: {loss.item()}\n")

            model_dir = os.path.join(root, "models")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), os.path.join(model_dir, f"model_{epoch+1}.pt"))
            print(f"Model saved as model_{epoch+1}.pt")

            results_dir = os.path.join(root, "results", f"epoch_{epoch+1}")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            x, y, noise, noise_ratio = next(iter(test_dataloader))
            x = x.to(device)
            y = y.to(device)
            noise = noise.to(device)

            x_waveform = istft_transform(x)
            y_waveform = istft_transform(y)

            pred_noise_ratio = model(x[..., 0])
            pred_noise = pred_noise_ratio * x[..., 0]
            pred_noise_waveform = istft_transform(torch.stack([pred_noise, x[..., 1]], dim=-1))

            denoised = F.relu(x[..., 0] - pred_noise)
            denoised_waveform = istft_transform(torch.stack([denoised, x[..., 1]], dim=-1))

            noise = istft_transform(noise)

            for i in range(min(BATCH_SIZE, 5)):
                try:
                    torchaudio.save(os.path.join(results_dir, f"{i}_input.wav"), x_waveform.cpu()[i : i + 1], 16000)
                    torchaudio.save(os.path.join(results_dir, f"{i}_target.wav"), y_waveform.cpu()[i : i + 1], 16000)
                    torchaudio.save(os.path.join(results_dir, f"{i}_output.wav"), pred_noise_waveform.cpu()[i : i + 1], 16000)
                    torchaudio.save(os.path.join(results_dir, f"{i}_denoised.wav"), denoised_waveform.cpu()[i : i + 1], 16000)
                    torchaudio.save(os.path.join(results_dir, f"{i}_noise.wav"), noise.cpu()[i : i + 1], 16000)
                    save_ratio_image(noise_ratio.cpu()[i : i + 1], os.path.join(results_dir, f"{i}_noise_ratio.png"))
                    save_ratio_image(pred_noise_ratio.cpu()[i : i + 1], os.path.join(results_dir, f"{i}_pred_noise_ratio.png"))
                except Exception as e:
                    print(f"Error saving samples: {e}")
            print(f"Sample results saved for epoch {epoch+1}")

        log_file.write("-" * 80 + "\n")


def collate_fn(batch):
    x, y, noise, noise_ratio = zip(*batch)
    max_length = max([x_.shape[-2] for x_ in x])
    xs = []
    ys = []
    noises = []
    noise_ratios = []
    for x_, y_, noise_, noise_ratio_ in zip(x, y, noise, noise_ratio):
        x_ = F.pad(x_, (0, 0, max_length - x_.shape[-2], 0))
        y_ = F.pad(y_, (0, 0, max_length - y_.shape[-2], 0))
        noise_ = F.pad(noise_, (0, 0, max_length - noise_.shape[-2], 0))
        noise_ratio_ = F.pad(noise_ratio_, (0, max_length - noise_ratio_.shape[-1]))
        xs.append(x_)
        ys.append(y_)
        noises.append(noise_)
        noise_ratios.append(noise_ratio_)
    x = torch.stack(xs)
    y = torch.stack(ys)
    noise = torch.stack(noises)
    noise_ratio = torch.stack(noise_ratios)
    return x, y, noise, noise_ratio


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
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=1000)
