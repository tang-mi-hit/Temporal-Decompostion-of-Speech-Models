import pytest
import torch
from transformers import (
    Data2VecAudioConfig,
    Data2VecAudioModel,
    HubertConfig,
    HubertModel,
    Wav2Vec2Config,
    Wav2Vec2Model,
    WavLMConfig,
    WavLMModel,
    WhisperConfig,
    WhisperModel,
)


@pytest.mark.parametrize(
    ("config_class", "model_class"),
    [
        (HubertConfig, HubertModel),
        (Wav2Vec2Config, Wav2Vec2Model),
        (WavLMConfig, WavLMModel),
        (Data2VecAudioConfig, Data2VecAudioModel),
    ],
)
def test_short_waveform_hidden_state_contract(config_class, model_class):
    config = config_class(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        conv_dim=(8,),
        conv_stride=(2,),
        conv_kernel=(4,),
        num_conv_pos_embedding_groups=2,
        num_conv_pos_embeddings=4,
    )
    model = model_class(config).eval()
    with torch.inference_mode():
        output = model(
            torch.zeros((1, 64)),
            output_hidden_states=True,
            return_dict=True,
        )
    assert len(output.hidden_states) == 2
    assert output.hidden_states[0].shape[0] == 1
    assert output.hidden_states[0].shape[2] == 8


def test_short_whisper_encoder_hidden_state_contract():
    config = WhisperConfig(
        vocab_size=32,
        num_mel_bins=80,
        d_model=8,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=16,
        max_source_positions=20,
        max_target_positions=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    encoder = WhisperModel(config).get_encoder().eval()
    with torch.inference_mode():
        output = encoder(
            torch.zeros((1, 80, 40)),
            output_hidden_states=True,
            return_dict=True,
        )
    assert len(output.hidden_states) == 2
    assert output.hidden_states[0].shape == (1, 20, 8)

