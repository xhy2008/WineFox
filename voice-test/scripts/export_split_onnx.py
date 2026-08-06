"""Export Kokoro split encoder/decoder ONNX models from PyTorch.

The C++ inference engine uses split mode: separate encoder (BERT + duration
predictor + text encoder + F0Ntrain) and decoder (iSTFTNet) ONNX sessions.

This script wraps the PyTorch KModel into two exportable sub-modules:
  - Encoder: input_ids, ref_s, speed → asr, F0_pred, N_pred, style_dec, pred_dur
  - Decoder: asr, F0_pred, N_pred, style_dec → audio
"""
import torch
import numpy as np
import os

OUTPUT_DIR = r'e:\winefox\voice-test\models'


class KokoroEncoder(torch.nn.Module):
    """Wrapper for Kokoro encoder: produces decoder inputs."""

    def __init__(self, model):
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.predictor = model.predictor
        self.text_encoder = model.text_encoder

    def forward(self, input_ids, ref_s, speed):
        input_lengths = torch.full(
            (input_ids.shape[0],), input_ids.shape[-1],
            device=input_ids.device, dtype=torch.long
        )
        text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(
            input_lengths.shape[0], -1
        ).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

        # BERT encoder
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)

        # Prosody style (back 128)
        s = ref_s[:, 128:]

        # Duration predictor
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        indices = torch.repeat_interleave(
            torch.arange(input_ids.shape[1], device=input_ids.device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (input_ids.shape[1], indices.shape[0]), device=input_ids.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)

        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)

        # Text encoder
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg

        # Acoustic style (front 128)
        style_dec = ref_s[:, :128]

        return asr, F0_pred, N_pred, style_dec, pred_dur


class KokoroDecoder(torch.nn.Module):
    """Wrapper for Kokoro decoder: produces audio."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, asr, F0_pred, N_pred, style_dec):
        audio = self.decoder(asr, F0_pred, N_pred, style_dec).squeeze()
        return audio


def main():
    print("Loading Kokoro PyTorch model...")
    from kokoro import KModel
    model = KModel(disable_complex=True).eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Dummy inputs
    batch_size = 1
    seq_len = 20  # token length
    n_frames = 100  # approximate frame count after duration expansion

    input_ids = torch.randint(1, 178, (batch_size, seq_len), dtype=torch.long)
    input_ids[:, 0] = 0   # BOS
    input_ids[:, -1] = 0  # EOS
    ref_s = torch.randn(batch_size, 256)
    speed = torch.tensor([1.0], dtype=torch.float32)

    # Export encoder
    print("Exporting encoder...")
    enc = KokoroEncoder(model).eval()
    enc_path = os.path.join(OUTPUT_DIR, 'kokoro-encoder.onnx')

    # Use dynamic axes for variable-length input
    torch.onnx.export(
        enc,
        (input_ids, ref_s, speed),
        enc_path,
        input_names=['input_ids', 'ref_s', 'speed'],
        output_names=['asr', 'F0_pred', 'N_pred', 'style_dec', 'pred_dur'],
        dynamic_axes={
            'input_ids': {1: 'seq_len'},
            'ref_s': {0: 'batch'},
            'asr': {2: 'n_frames'},
            'F0_pred': {1: 'n_frames2'},
            'N_pred': {1: 'n_frames2'},
            'pred_dur': {1: 'seq_len'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Encoder: {enc_path} ({os.path.getsize(enc_path)/1e6:.1f} MB)")

    # Export decoder
    print("Exporting decoder...")
    dec = KokoroDecoder(model).eval()

    # Run encoder once to get real decoder input shapes
    with torch.no_grad():
        asr, F0_pred, N_pred, style_dec, _ = enc(input_ids, ref_s, speed)

    dec_path = os.path.join(OUTPUT_DIR, 'kokoro-decoder.onnx')
    torch.onnx.export(
        dec,
        (asr, F0_pred, N_pred, style_dec),
        dec_path,
        input_names=['asr', 'F0_pred', 'N_pred', 'style_dec'],
        output_names=['audio'],
        dynamic_axes={
            'asr': {2: 'n_frames'},
            'F0_pred': {1: 'n_frames2'},
            'N_pred': {1: 'n_frames2'},
            'audio': {0: 'audio_len'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Decoder: {dec_path} ({os.path.getsize(dec_path)/1e6:.1f} MB)")

    # Quick sanity check: load and run in ORT
    print("\nSanity check with ORT...")
    import onnxruntime as ort

    sess_enc = ort.InferenceSession(enc_path, providers=['CPUExecutionProvider'])
    sess_dec = ort.InferenceSession(dec_path, providers=['CPUExecutionProvider'])

    enc_out = sess_enc.run(
        None,
        {'input_ids': input_ids.numpy(),
         'ref_s': ref_s.numpy(),
         'speed': speed.numpy()}
    )
    dec_out = sess_dec.run(
        None,
        {'asr': enc_out[0], 'F0_pred': enc_out[1],
         'N_pred': enc_out[2], 'style_dec': enc_out[3]}
    )
    audio = dec_out[0]
    print(f"  Audio shape: {audio.shape}")
    print(f"  Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
    print(f"  Audio duration: {len(audio[0])/24000:.2f}s")
    print("\nDone!")


if __name__ == '__main__':
    main()
