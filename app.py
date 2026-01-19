#Importing internal and external libraries and functions.

import os
import gradio as gr
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from api.inference import run_inference
from api.postprocess import create_overlay, extract_binary_mask
from api.utils import masks_from_results
from api.metrics import evaluate_instance_segmentation



#Here, we loading example images and gt masks for screens. You have to change image path according to your file distribution. Png extention for gt mask.

EXAMPLES = [
    ["ADD YOUR PATH/00d6e742-756c-4f99-96b6-4f47730ef944.jpg", "ADD YOUR PATH/00d6e742-756c-4f99-96b6-4f47730ef944.png"],
    ["ADD YOUR PATH/0bc33d71-f7b7-491a-919b-97026420060b.jpeg", "ADD YOUR PATH/0bc33d71-f7b7-491a-919b-97026420060b.png"],
    ["ADD YOUR PATH/cjvnxa4wep8as0866zqd41859.jpeg", "ADD YOUR PATH/cjvnxa4wep8as0866zqd41859.png"],
]




# =========================
# BACKEND FUNCTIONS
# =========================

def run_inference_ui(image, model, conf):
    if image is None:
        return None, None, "❌ Please upload an image."

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = run_inference(
        image_bgr,
        model_name=model,
        conf=conf
    )

    masks = masks_from_results(results)

    overlay = create_overlay(image_bgr, masks)
    binary_mask = extract_binary_mask(masks, image_bgr.shape)

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return overlay_rgb, binary_mask, "✅ Inference completed."


def run_evaluation_ui(image, gt_mask, model, iou_thr):
    if image is None or gt_mask is None:
        return "❌ Image and Ground Truth are required."

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = run_inference(image_bgr, model_name=model)
    pred_masks = masks_from_results(results)

    metrics = evaluate_instance_segmentation(
        gt_mask=gt_mask,          
        pred_masks=pred_masks,
        iou_threshold=iou_thr
    )

    table_md = f"""
### 📊 Evaluation Metrics

| Metric | Value |
|-------|-------|
| mAP@50 | {metrics['mAP50']:.3f} |
| Precision | {metrics['precision']:.3f} |
| Recall | {metrics['recall']:.3f} |
| TP | {metrics['TP']} |
| FP | {metrics['FP']} |
| FN | {metrics['FN']} |
"""

    return table_md



def run_comparison_ui(image, gt_mask, models, iou_thr):
    if image is None or gt_mask is None or not models:
        return None, None, "❌ Please provide image, GT mask and models."

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    rows = []

    for model in models:
        results = run_inference(image_bgr, model_name=model)
        pred_masks = masks_from_results(results)

        metrics = evaluate_instance_segmentation(
            gt_mask=gt_mask,
            pred_masks=pred_masks,
            iou_threshold=iou_thr
        )

        rows.append({
            "Model": model,
            "mAP@50": metrics["mAP50"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"]
        })

    df = pd.DataFrame(rows)

    # -------- Plot --------
    fig, ax = plt.subplots(figsize=(9, 5))

    df.set_index("Model")[["mAP@50", "Precision", "Recall"]].plot(
        kind="bar", ax=ax
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=20,
        ha="right"
    )
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    # -------- SAVE FIGURE (EKLENEN KISIM) --------
    os.makedirs("outputs", exist_ok=True)
    plot_path = "outputs/model_comparison.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    table_txt = df.to_string(index=False)

    return fig, plot_path, f"📋 **Comparison Results**\n\n```\n{table_txt}\n```"



# =========================
# CUSTOM CSS : You can change UI colors etc. from here.
# =========================

custom_css = """
/* ===== PAGE BACKGROUND ===== */
body, .gradio-container {
    background: linear-gradient(
        135deg,
        #f1f8f4 0%,
        #e6f4ea 40%,
        #eef7f1 100%
    ) !important;
}

/* ===== DESC FORMAT ===== */
.app-description {
    font-family: "Inter", "Segoe UI", "Roboto", sans-serif;
    font-size: 15.5px;
    line-height: 1.75;
    color: #2f4f2f;
    max-width: 900px;
    margin-top: 12px;
}

.app-description p {
    margin-bottom: 10px;
}

/* ===== MAIN APP CARD ===== */
#root {
    background: transparent !important;
}

/* ===== CARDS / PANELS ===== */
.gr-box,
.gr-panel,
.gr-form,
.gr-accordion,
.gr-tabitem {
    background: rgba(255, 255, 255, 0.85) !important;
    border-radius: 18px;
    border: 1px solid rgba(0, 0, 0, 0.06);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
}

/* ===== ACCORDION HEADER ===== */
.gr-accordion > summary {
    font-weight: 600;
    font-size: 15px;
    color: #2e7d32;
}

/* ===== BUTTONS ===== */
button {
    border-radius: 14px !important;
    font-weight: 600;
}

button.primary {
    background: linear-gradient(135deg, #2ecc71, #27ae60) !important;
    color: white !important;
}

/* ==============================
   IMAGE CONTAINERS – COLORIZED
   ============================== */

/* 🟣 Inference Input Image */
#input-image {
    background: #ffffff;
    border-radius: 18px;
    padding: 10px;
    border: 2px solid rgba(142, 68, 173, 0.45);
}
#input-image:hover {
    box-shadow: 0 0 18px rgba(142, 68, 173, 0.30);
}

/* 🔵 Overlay Image */
#overlay-image {
    background: #ffffff;
    border-radius: 18px;
    padding: 10px;
    border: 2px solid rgba(41, 128, 185, 0.45);
}
#overlay-image:hover {
    box-shadow: 0 0 18px rgba(41, 128, 185, 0.30);
}

/* 🌸 Prediction Mask */
#mask-image {
    background: #ffffff;
    border-radius: 18px;
    padding: 10px;
    border: 2px solid rgba(232, 67, 147, 0.45);
}
#mask-image:hover {
    box-shadow: 0 0 18px rgba(232, 67, 147, 0.30);
}

/* 🟠 Evaluation Input Image */
#eval-input-image {
    background: #ffffff;
    border-radius: 18px;
    padding: 10px;
    border: 2px solid rgba(230, 126, 34, 0.45);
}
#eval-input-image:hover {
    box-shadow: 0 0 18px rgba(230, 126, 34, 0.30);
}

/* 🔴 Ground Truth Mask */
#eval-gt-mask {
    background: #ffffff;
    border-radius: 18px;
    padding: 10px;
    border: 2px solid rgba(192, 57, 43, 0.45);
}
#eval-gt-mask:hover {
    box-shadow: 0 0 18px rgba(192, 57, 43, 0.30);
}

/* 📈 Model Comparison Graph */
/* 🟥 Model Comparison – Input Image */
#cmp-image {
    background: #ffffff;
    border-radius: 18px;
    padding: 10px;
    border: 2px solid rgba(155, 89, 182, 0.45);
}
#cmp-image:hover {
    box-shadow: 0 0 18px rgba(155, 89, 182, 0.30);
}

/* 🟧 Model Comparison – Ground Truth Mask */
#cmp-mask {
    background: #ffffff;
    border-radius: 18px;
    padding: 10px;
    border: 2px solid rgba(231, 76, 60, 0.45);
}
#cmp-mask:hover {
    box-shadow: 0 0 18px rgba(231, 76, 60, 0.30);
}

/* ===== MARKDOWN HEADERS ===== */
h1, h2, h3 {
    color: #1b5e20;
}

/* ===== EXTERNAL LINKS ===== */
.fa-github:hover,
.fa-pen-nib:hover {
    color: #1b5e20;
    transform: scale(1.15);
    transition: 0.2s ease;
}

/* ===== DISTRIBUTE TABS: LEFT / CENTER / RIGHT ===== */
.gr-tabs {
    justify-content: space-between !important;
    width: 100%;
    padding: 5 80px;  /* kenarlardan boşluk */
}


"""


# =========================
# UI
# =========================

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="green"),
    css=custom_css,
    title="GenYOLO-Leaf | Interactive Segmentation"
) as demo:

    gr.HTML("""
    <link rel="stylesheet"
          href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    """)

    
    gr.HTML("""
<div style="
    position: absolute;
    top: 18px;
    right: 20px;
    display: flex;
    flex-direction: column;   /* ALT ALTA */
    gap: 10px;
    font-size: 12px;
    font-weight: 600;
    z-index: 1000;
">
    <!-- GitHub -->
    <a href="https://github.com/aaslihanyildirim/GenYOLO-Leaf"
       target="_blank"
       title="Official GitHub Repository"
       style="
            display: flex;
            align-items: center;
            gap: 8px;
            color: #2e7d32;
            text-decoration: none;
       ">
        <i class="fab fa-github" style="font-size:30px;"></i>
        <span>Official Repository</span>
    </a>

    <!-- Paper -->
    <a href="#"
       title="Research Paper (Coming Soon)"
       style="
            display: flex;
            align-items: center;
            gap: 2px;
            color: #2e7d32;
            text-decoration: none;
            cursor: default;
       ">
        <span style="font-size:24px;">🖋️</span>
        <span>Research Paper (Comming Soon)</span>
    </a>
</div>
""")

 

    gr.HTML("""
<div style="text-align:center">
    <h1>🍀 GenYOLO-Leaf Interactive Platform 🍃</h1>
    <p><b>Inference, Evaluation & Model Comparison for Leaf Instance Segmentation</b></p>
</div>
""")
    
    gr.Markdown("""
---
<div class="app-description" style="
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 300px;          /* Sütunlar arası mesafe */
    max-width: 1500px;
    margin: 0;
    font-size: 17px;
    line-height: 1.6;
">

<!-- LEFT COLUMN -->
<div style="text-align: left;">
<b style="display:block; margin-bottom: 16px;">📘 About this platform</b>


<ul>
  <li>Run instance segmentation inference on leaf images with data-centric models</li>
  <li>Evaluate predictions with instance-level metrics</li>
  <li>Compare multiple models under identical conditions</li>
</ul>
</div>

<!-- RIGHT COLUMN -->
<div style="text-align: left;">
<b style="display:block; margin-bottom: 16px;">⚠️ Pay Attention</b>


<ul>
  <li>Ground truth masks can be uploaded in <b>any resolution</b> and <b>any image format</b></li>
  <li>Please select a maximum of <b>3 models</b> on the model comparison screen, as this will increase the processing time</li>
  <li>If you want to use this work and the developed weights, remember to reference the GitHub repository and the original article</li>
</ul>
</div>

</div>
""")

    with gr.Tabs():

        # =========================
        # INFERENCE TAB
        # =========================
        with gr.Tab("🔍 Inference"):
            with gr.Row():
                with gr.Column():
                    inf_image = gr.Image(
                        label="Input Image",
                        type="numpy",
                        height=300,
                        elem_id="input-image"
                    )

                    gr.Examples(
                    examples=EXAMPLES,
                    inputs=[inf_image],
                    label="📂 Try with Example Images"
                    )


                    inf_model = gr.Dropdown(
                        label="Model",
                        choices=["GenYOLO-Leaf-V8-N", "GenYOLO-Leaf-V8-S", "GenYOLO-Leaf-V8-M","GenYOLO-Leaf-V8-L",
                "GenYOLO-Leaf-V8-X","GenYOLO-Leaf-V11-N","GenYOLO-Leaf-V11-S","GenYOLO-Leaf-V11-M","GenYOLO-Leaf-V11-L","GenYOLO-Leaf-V11-X"],
                value="GenYOLO-Leaf-V8-N"
                   )

                    inf_conf = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold"
                    )


                    inf_btn = gr.Button("🌱 Run Inference", variant="primary")
                    inf_status = gr.Markdown("🟢 Ready")

                with gr.Column():
                    overlay_out = gr.Image(
                        label="Instance Segmentation Overlay",
                        format="png", 
                        height=250,
                        elem_id="overlay-image"
                    )

                    mask_out = gr.Image(
                        label="Binary Mask",
                        format="png",
                        height=250,
                        elem_id="mask-image"
                    )

        # =========================
        # EVALUATION TAB
        # =========================
        with gr.Tab("📊 Evaluation"):

    # ---- ÜST: Görseller yan yana ----
          with gr.Row():
           with gr.Column(scale=1):
             eval_image = gr.Image(
             label="Input Image",
             type="numpy",
             height=280,
             elem_id="eval-input-image"
        )
           gt_mask=gr.Image(
             label="Ground Truth Mask",
             type="numpy",
             image_mode="RGB",
             height=280,
             elem_id="eval-gt-mask"
        )
           
           gr.Examples(
             examples=EXAMPLES,
             inputs=[eval_image, gt_mask],
             label="📂 Evaluation Examples"
        )   
          

    # ---- ALT: Kontroller + metrik ----
          with gr.Row():
            with gr.Column(scale=1):
             eval_model = gr.Dropdown(
                label="Model",
                choices=["GenYOLO-Leaf-V8-N", "GenYOLO-Leaf-V8-S", "GenYOLO-Leaf-V8-M","GenYOLO-Leaf-V8-L",
                "GenYOLO-Leaf-V8-X","GenYOLO-Leaf-V11-N","GenYOLO-Leaf-V11-S","GenYOLO-Leaf-V11-M","GenYOLO-Leaf-V11-L","GenYOLO-Leaf-V11-X"],
                value="GenYOLO-Leaf-V8-N"
            )

            eval_iou = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                step=0.05,
                value=0.5,
                label="IoU Threshold"
            )
            eval_btn = gr.Button("📊 Run Evaluation", variant="primary")

            with gr.Column(scale=1):
             metrics_md = gr.Markdown(
                "📊 **Evaluation Metrics**\n\nWaiting for evaluation...",
                elem_id="metrics-card"
            )
        # =========================
        # MODEL COMPARISON TAB
        # =========================
        with gr.Tab("📈 Model Comparison"):
            with gr.Row():
                with gr.Column():
                    cmp_image = gr.Image(
                        label="Input Image",
                        type="numpy",
                        height=300,
                        elem_id="cmp-image"
                    )

                    cmp_gt = gr.Image(
                        label="Ground Truth Mask",
                        type="numpy",
                        image_mode="RGB",
                        height=300,
                        elem_id="cmp-mask"
                    )

                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=[cmp_image, cmp_gt],
                        label="📂 Comparison Examples"
                    )

                    cmp_models = gr.CheckboxGroup(
                        choices=["GenYOLO-Leaf-V8-N", "GenYOLO-Leaf-V8-S", "GenYOLO-Leaf-V8-M","GenYOLO-Leaf-V8-L",
                "GenYOLO-Leaf-V8-X","GenYOLO-Leaf-V11-N","GenYOLO-Leaf-V11-S","GenYOLO-Leaf-V11-M","GenYOLO-Leaf-V11-L","GenYOLO-Leaf-V11-X"],
                        value="GenYOLO-Leaf-V8-N",
                        label="Models to Compare"
                    )

                    cmp_iou = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="IoU Threshold"
                    )


                    cmp_btn = gr.Button("📈 Run Comparison", variant="primary")

                with gr.Column():
                    cmp_plot = gr.Plot(label="Model Comparison",
                    elem_id="cmp-graph")
                    cmp_table = gr.Markdown(
                        "📋 **Comparison Results**\n\nWaiting for comparison...",
                        elem_id="metrics-card"
                    )

    # =========================
    # EVENTS
    # =========================

    inf_btn.click(
        fn=run_inference_ui,
        inputs=[inf_image, inf_model, inf_conf],
        outputs=[overlay_out, mask_out, inf_status]
    )

    eval_btn.click(
        fn=run_evaluation_ui,
        inputs=[eval_image, gt_mask, eval_model, eval_iou],
        outputs=metrics_md
    )

    cmp_btn.click(
        fn=run_comparison_ui,
        inputs=[cmp_image, cmp_gt, cmp_models, cmp_iou],
        outputs=[cmp_plot, cmp_table]
    )


demo.launch()

