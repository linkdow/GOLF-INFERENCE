import torch
import os
import numpy as np
import yaml
import matplotlib.pyplot as plt
import torchaudio
import pytorch_lightning as pl
import sys
sys.path.append("/home/linkdow/svs/nnsvs/golf/") # fix if error on import other functions

from ltng.vocoder import DDSPVocoder
from IPython.display import Audio
import librosa
from librosa.display import specshow
from pathlib import Path
from tqdm import tqdm

from models.utils import fir_filt, linear_upsample, TimeContext, smooth_phase_offset, get_transformed_lf
from infer import convert2samplewise
from test_rtf import dict2object, get_instance

def generate_audio(spectrogram,config_path,device="cuda"):

    # put these variables as function's parameter later if necessary :
    device = "cuda"
    lr = 0.001
    iterations = 1000
    phase_offset_hop_length = 1200

    print(type(spectrogram[0][0]))
    spectrogram = spectrogram.astype(np.float32)
    spectrogram = torch.tensor(spectrogram, device=device).unsqueeze(0) # unsqueeze to create a batch of one
    print("Spectrogram shape:", spectrogram.shape)
    print(type(spectrogram))
    print("Loading config file:", config_path)
    # Load the config file
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_configs = config["model"]

    sr = model_configs["sample_rate"]

    model_configs["feature_trsfm"]["init_args"]["sample_rate"] = model_configs["sample_rate"]
    model_configs["feature_trsfm"]["init_args"]["window"] = model_configs["window"]
    model_configs["feature_trsfm"]["init_args"]["hop_length"] = model_configs["hop_length"]
    # print(model_configs)
    model_configs = dict2object(model_configs)
    
    ckpt_path = config["ckpt_path"]

    model_configs = dict2object(model_configs)

    
    print("Loading model from checkpoint:", ckpt_path)
    model = DDSPVocoder.load_from_checkpoint(ckpt_path, **model_configs).to(device)
    model.eval()

    # Transform the input spectrogram to the model's input format
    print("Transforming input into the model's input format")
    with torch.no_grad():
        feats = model.feature_trsfm(spectrogram)
        # je suis bloqué ici for now
        (
            f0_params,
            harm_osc_params,
            harm_filt_params,
            noise_filt_params,
            noise_params,
        ) = model.encoder(feats)
        f0_hat, *voicing_param = f0_params
        phase = f0_hat / sr
        ctx = TimeContext(120)
        upsampled_phase = linear_upsample(phase, ctx)
        if len(voicing_param) > 0:
            voicing_logits = voicing_param[0]
            voicing = voicing_logits.sigmoid()
            upsampled_voicing = linear_upsample(voicing, ctx)
            upsampled_phase = upsampled_phase * upsampled_voicing

        phase_offsets = torch.nn.Parameter(
            torch.rand(1, spectrogram.shape[1] // phase_offset_hop_length + 1, device=device)
        )

        optimizer = torch.optim.Adam([phase_offsets], lr=lr)

        decoder = model.decoder

        def forward(offsets):
            upsampled_offsets = linear_upsample(smooth_phase_offset(offsets), TimeContext(phase_offset_hop_length))
            upsampled_offsets = upsampled_offsets[:, : upsampled_phase.shape[1]]
            harm_osc = decoder.harm_oscillator(upsampled_phase[:, : upsampled_offsets.shape[1]], *harm_osc_params, ctx=ctx, upsampled_phase_offset=upsampled_offsets)
            noise = decoder.noise_generator(harm_osc, *noise_params, ctx=ctx)
            if decoder.harm_filter is not None:
                harm_osc = decoder.harm_filter(harm_osc, *harm_filt_params, ctx=ctx)
            if decoder.noise_filter is not None:
                noise = decoder.noise_filter(noise, *noise_filt_params, ctx=ctx)

            out = harm_osc[:, : noise.shape[1]] + noise[:, : harm_osc.shape[1]]

            if decoder.end_filter is not None:
                return decoder.end_filter(out)
            else:
                return out
        
        # one shot inference
        print("Performing one shot inference")
        with torch.no_grad():
            y_zero = forward(phase_offsets)
        
        return y_zero
        
        # # save the output audio
        # torchaudio.save("test1.wav", y_zero.detach().cpu(), sr)
        # print("Output audio saved as test1.wav")
        
        # # better inference
        # pbar = tqdm(range(iterations))

        # for i in pbar:
        #     optimizer.zero_grad()
        #     y_hat = forward(phase_offsets)
        #     loss = torch.nn.functional.mse_loss(y[:, : y_hat.shape[1]], y_hat[:, : y.shape[1]])
        #     loss.backward()
        #     optimizer.step()
        #     pbar.set_description(f"Loss: {loss.item() * min(y.shape[1], y_hat.shape[1])}")
        
        # # save the output audio
        # torchaudio.save("test1.wav", y_hat.detach().cpu(), sr)
