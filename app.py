"""
DIP-UP – Gradio Web Interface
=====================================
Web UI for DIP-UP MRI phase unwrapping.

Launch:
    python app.py                   # CPU
    python app.py --share           # public Gradio link
    python app.py --server-port 8080

Docker:
    docker compose up               # see docker-compose.yml
"""

import argparse
import os
import tempfile
import traceback

import gradio as gr
import nibabel as nib
import numpy as np

from inference import run_dipup


def _make_slice_figure(nii_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vol = nib.load(nii_path).get_fdata(dtype=np.float32)
    vmin, vmax = np.percentile(vol, [2, 98])
    vol_n = np.clip((vol - vmin) / max(vmax - vmin, 1e-6), 0, 1)

    slices = {
        "Axial":    vol_n[:, :, vol_n.shape[2] // 2].T,
        "Coronal":  vol_n[:, vol_n.shape[1] // 2, :].T,
        "Sagittal": vol_n[vol_n.shape[0] // 2, :, :].T,
    }

    imgs = []
    for title, sl in slices.items():
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(sl, cmap="gray", origin="lower", aspect="equal")
        ax.set_title(title, fontsize=12)
        ax.axis("off")
        fig.tight_layout(pad=0.5)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        imgs.append(buf[:, :, :3].copy())
        plt.close(fig)

    return imgs[0], imgs[1], imgs[2]


def reconstruct(
    phase_file,
    mask_file,
    checkpoint_file,
    n_iter,
    lr,
    progress=gr.Progress(track_tqdm=True),
):
    if phase_file is None:
        raise gr.Error("Please upload a wrapped phase NIfTI file.")

    checkpoint_path = checkpoint_file.name if checkpoint_file else None
    output_dir = tempfile.mkdtemp(prefix="dipup_out_")

    def _progress(frac, msg):
        progress(frac, desc=msg)

    try:
        unwrapped_path = run_dipup(
            phase_nii_path=phase_file.name,
            mask_nii_path=mask_file.name if mask_file else None,
            checkpoint_path=checkpoint_path,
            n_iter=int(n_iter),
            lr=float(lr),
            output_dir=output_dir,
            progress_fn=_progress,
        )
    except Exception:
        raise gr.Error(
            "Unwrapping failed. Check the log for details.\n\n"
            + traceback.format_exc()
        )

    try:
        ax_img, cor_img, sag_img = _make_slice_figure(unwrapped_path)
    except Exception:
        ax_img = cor_img = sag_img = None

    status = "✅ Phase unwrapping complete! Download the result below."
    return status, unwrapped_path, ax_img, cor_img, sag_img


TITLE = "DIP-UP – Deep Image Prior Phase Unwrapping"
DESCRIPTION = """
**MRI Phase Unwrapping** using a Deep Image Prior approach (*DIP-UP / PhaseNet3D*)
([paper](https://doi.org/10.1002/mrm.28457)).

DIP-UP optimises a network on your data at inference time — no general checkpoint needed
(though a pretrained network can be provided for faster convergence).

**Quick-start:**
1. Upload your wrapped phase NIfTI (values in radians).
2. Optionally upload a brain mask and/or a pretrained checkpoint.
3. Adjust the number of iterations (more = better, slower).
4. Click **Run Unwrapping**.
"""


def build_ui():
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                phase_file = gr.File(
                    label="Wrapped phase NIfTI (.nii / .nii.gz, radians)",
                    file_types=[".nii", ".gz"],
                )
                mask_file = gr.File(
                    label="Brain mask NIfTI (optional)",
                    file_types=[".nii", ".gz"],
                )
                checkpoint_file = gr.File(
                    label="Pretrained checkpoint .pth (optional — random init if omitted)",
                    file_types=[".pth"],
                )

                gr.Markdown("### Optimisation parameters")
                n_iter = gr.Slider(
                    label="Number of iterations",
                    minimum=200,
                    maximum=5000,
                    step=100,
                    value=2000,
                )
                lr = gr.Number(
                    label="Learning rate",
                    value=1e-6,
                    minimum=1e-9,
                    maximum=1e-3,
                    step=1e-7,
                )

                run_btn = gr.Button("▶ Run Unwrapping", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Results")
                status_box = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False,
                    placeholder="Unwrapping output will appear here …",
                )
                download_file = gr.File(label="⬇ Download Unwrapped Phase NIfTI")

                gr.Markdown("#### Preview (middle slice)")
                with gr.Row():
                    axial_img    = gr.Image(label="Axial",    show_label=True)
                    coronal_img  = gr.Image(label="Coronal",  show_label=True)
                    sagittal_img = gr.Image(label="Sagittal", show_label=True)

        run_btn.click(
            fn=reconstruct,
            inputs=[phase_file, mask_file, checkpoint_file, n_iter, lr],
            outputs=[status_box, download_file, axial_img, coronal_img, sagittal_img],
        )

        gr.Markdown(
            "---\n"
            "**Citation:** Zhu X, et al. *Phase Unwrapping with Deep Image Prior.* "
            "Magnetic Resonance in Medicine, 2021. "
            "[doi:10.1002/mrm.28457](https://doi.org/10.1002/mrm.28457)\n\n"
            "**Source code:** [github.com/sunhongfu/DIP-UP](https://github.com/sunhongfu/DIP-UP)"
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIP-UP Gradio server")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
        allowed_paths=[tempfile.gettempdir()],
    )
