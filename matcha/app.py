import tempfile
from argparse import Namespace
from pathlib import Path

import gradio as gr
import soundfile as sf
import torch

from matcha.cli import (
    MATCHA_URLS,
    VOCODER_URLS,
    assert_model_downloaded,
    get_device,
    load_matcha,
    load_vocoder,
    process_text,
    to_waveform,
)
from matcha.utils.utils import get_user_data_dir, plot_tensor

LOCATION = Path(get_user_data_dir())

args = Namespace(
    cpu=True,
    model="akyl_ai",
    vocoder="hifigan_T2_v1",
)

CURRENTLY_LOADED_MODEL = args.model


def MATCHA_TTS_LOC(x):
    return LOCATION / f"{x}.ckpt"


def VOCODER_LOC(x):
    return LOCATION / f"{x}"


LOGO_URL = "https://github.com/simonlobgromov/Matcha-TTS/blob/main/photo_2024-04-07_15-59-52.png"
RADIO_OPTIONS = {

    "Akyl_AI": {
        "model": "akyl_ai",
        "vocoder": "hifigan_T2_v1",
    },
}

# Ensure all the required models are downloaded
assert_model_downloaded(MATCHA_TTS_LOC("akyl_ai"), MATCHA_URLS["akyl_ai"])
assert_model_downloaded(VOCODER_LOC("hifigan_T2_v1"), VOCODER_URLS["hifigan_T2_v1"])


device = get_device(args)

# Load default model
model = load_matcha(args.model, MATCHA_TTS_LOC(args.model), device)
vocoder, denoiser = load_vocoder(args.vocoder, VOCODER_LOC(args.vocoder), device)


def load_model(model_name, vocoder_name):
    model = load_matcha(model_name, MATCHA_TTS_LOC(model_name), device)
    vocoder, denoiser = load_vocoder(vocoder_name, VOCODER_LOC(vocoder_name), device)
    return model, vocoder, denoiser


def load_model_ui(model_type, textbox):
    model_name, vocoder_name = RADIO_OPTIONS[model_type]["model"], RADIO_OPTIONS[model_type]["vocoder"]

    global model, vocoder, denoiser, CURRENTLY_LOADED_MODEL  # pylint: disable=global-statement
    if CURRENTLY_LOADED_MODEL != model_name:
        model, vocoder, denoiser = load_model(model_name, vocoder_name)
        CURRENTLY_LOADED_MODEL = model_name

    if model_name == "akyl_ai":
        single_speaker_examples = gr.update(visible=True)
        multi_speaker_examples = gr.update(visible=False)
        length_scale = gr.update(value=0.95)
    else:
        single_speaker_examples = gr.update(visible=False)
        multi_speaker_examples = gr.update(visible=True)
        length_scale = gr.update(value=0.85)

    return (
        textbox,
        gr.update(interactive=True),
        single_speaker_examples,
        multi_speaker_examples,
        length_scale,
    )


@torch.inference_mode()
def process_text_gradio(text):
    output = process_text(1, text, device)
    return output["x_phones"][1::2], output["x"], output["x_lengths"]


@torch.inference_mode()
def synthesise_mel(text, text_length, n_timesteps, temperature, length_scale, spk=-1):
    spk = torch.tensor([spk], device=device, dtype=torch.long) if spk >= 0 else None
    output = model.synthesise(
        text,
        text_length,
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spk,
        length_scale=length_scale,
    )
    output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        sf.write(fp.name, output["waveform"], 22050, "PCM_24")

    return fp.name, plot_tensor(output["mel"].squeeze().cpu().numpy())


def ljspeech_example_cacher(text, n_timesteps, mel_temp, length_scale, spk=-1):
    global CURRENTLY_LOADED_MODEL  # pylint: disable=global-statement
    if CURRENTLY_LOADED_MODEL == "akyl_ai":
        global model, vocoder, denoiser  # pylint: disable=global-statement
        model, vocoder, denoiser = load_model("akyl_ai", "hifigan_T2_v1")
        CURRENTLY_LOADED_MODEL = "akyl_ai"

    phones, text, text_lengths = process_text_gradio(text)
    audio, mel_spectrogram = synthesise_mel(text, text_lengths, n_timesteps, mel_temp, length_scale, spk)
    return phones, audio, mel_spectrogram


def main():
    description = """# AkylAI TTS Mini"""

    with gr.Blocks(title="AkylAI TTS") as demo:
        processed_text = gr.State(value=None)
        processed_text_len = gr.State(value=None)

        with gr.Box():
            with gr.Row():
                gr.Markdown(description, scale=3)
            with gr.Row():
                image_url = "https://github.com/simonlobgromov/Matcha-TTS/blob/main/photo_2024-04-07_15-59-52.png?raw=true"
                gr.Image(image_url, label=None, width=660, height=315, show_label=False)

        with gr.Box():
            radio_options = list(RADIO_OPTIONS.keys())
            model_type = gr.Radio(
                radio_options, value=radio_options[0], label="Choose a Model", interactive=True, container=False, visible=False,
            )

            with gr.Row():
                gr.Markdown("## Текстти кыргыз тилинде жазыңыз\n### Text Input")
            with gr.Row():
                text = gr.Textbox(value="", label=None, scale=3, show_label=False)

            with gr.Row():
                gr.Markdown("## Сүйлөө ылдамдыгы\n### Speaking rate")
                # gr.Markdown("")
                
            with gr.Row():
                n_timesteps = gr.Slider(
                    label="Number of ODE steps",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=10,
                    interactive=True,
                    visible=False
                )
                length_scale = gr.Slider(
                    label=None,
                    minimum=0.5,
                    maximum=1,
                    step=0.05,
                    value=0.9,
                    interactive=True,
                    show_label=False
                )
                mel_temp = gr.Slider(
                    label="Sampling temperature",
                    minimum=0.00,
                    maximum=2.001,
                    step=0.16675,
                    value=0.667,
                    interactive=True,
                    visible=False
                )

                synth_btn = gr.Button("БАШТОО | RUN")


        phonetised_text = gr.Textbox(interactive=False, scale=10, label=None, visible=False )

        with gr.Box():
            with gr.Row():
                mel_spectrogram = gr.Image(interactive=False, label="mel spectrogram", visible=False)

                # with gr.Row():
                audio = gr.Audio(interactive=False, label="Audio")

        with gr.Row(visible=True) as example_row_lj_speech:
            examples = gr.Examples(  # pylint: disable=unused-variable
                examples=[
                    [
                        "Баарыңарга салам, менин атым Акылай. Мен бардыгын бул жерде Инновация борборунда көргөнүмө абдан кубанычтамын.",
                        50,
                        0.677,
                        0.95,
                    ],
                    [
                        "Мага колдоо көрсөтүп, мени тандагандарга ыраазымын. Айыл үчүн иштейбиз, жол курабыз, асфальт төшөйбүз”, — деген ал.",
                        2,
                        0.677,
                        0.95,
                    ],
 
                  
                ],
                fn=ljspeech_example_cacher,
                inputs=[text, n_timesteps, mel_temp, length_scale],
                outputs=[phonetised_text, audio, mel_spectrogram],
                cache_examples=True,
            )

      
        model_type.change(lambda x: gr.update(interactive=False), inputs=[synth_btn], outputs=[synth_btn]).then(
            load_model_ui,
            inputs=[model_type, text],
            outputs=[text, synth_btn, example_row_lj_speech, length_scale],
        )

        synth_btn.click(
            fn=process_text_gradio,
            inputs=[
                text,
            ],
            outputs=[phonetised_text, processed_text, processed_text_len],
            api_name="AkylAI TTS Mini",
            queue=True,
        ).then(
            fn=synthesise_mel,
            inputs=[processed_text, processed_text_len, n_timesteps, mel_temp, length_scale],
            outputs=[audio, mel_spectrogram],
        )

        demo.queue().launch()


if __name__ == "__main__":
    main()
