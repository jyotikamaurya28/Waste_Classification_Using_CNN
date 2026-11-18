import os
import time

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

import config

if not os.path.exists(config.MODEL_PATH):
    raise FileNotFoundError("best_model.keras not found. Run `python train.py` first.")
pred_model = load_model(config.MODEL_PATH)

CATEGORIES = {
    "recyclable": dict(
        emoji="♻️", title="Recyclable Waste", bin="Blue bin", color="#3b82f6",
        tips=["Rinse off food or liquid residue", "Keep it dry and uncontaminated",
              "Flatten boxes and bottles to save space"],
    ),
    "organic": dict(
        emoji="🍎", title="Organic Waste", bin="Green bin", color="#10b981",
        tips=["Remove plastic wrapping or stickers", "Ideal for composting",
              "Don't mix with recyclables"],
    ),
}


def preprocess(image):
    img = np.asarray(image)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    img = tf.image.resize(img, config.IMG_SIZE).numpy().astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def placeholder_card(msg="Upload a photo and I'll tell you which bin it belongs in."):
    return f"""
    <div class="card empty">
        <div class="big">🗑️</div>
        <h3>Ready to classify</h3>
        <p>{msg}</p>
        <div class="hints">
            <span>🍎 Food scraps</span><span>🍌 Fruit peels</span><span>🥬 Vegetables</span>
            <span class="b">🧴 Plastic bottles</span><span class="b">📦 Cardboard</span><span class="b">🥫 Metal cans</span>
        </div>
    </div>"""


def result_card(key, conf, secs):
    c = CATEGORIES[key]
    warn = ""
    if conf < config.LOW_CONF:
        warn = ('<div class="warn">⚠️ Low confidence. Retake with a single item, '
                'plain background and good lighting.</div>')
    tips = "".join(f"<li>{t}</li>" for t in c["tips"])
    return f"""
    <div class="card" style="--accent:{c['color']}">
        <div class="row">
            <div class="big">{c['emoji']}</div>
            <div><span class="badge">{c['bin']}</span><h2>{c['title']}</h2></div>
        </div>
        <div class="meta"><span>Confidence</span><b>{conf:.1%}</b></div>
        <div class="bar"><div style="width:{conf*100:.1f}%"></div></div>
        {warn}
        <h4>Disposal tips</h4>
        <ul>{tips}</ul>
        <div class="foot">Inference time: {secs*1000:.0f} ms</div>
    </div>"""


def stats_html(state):
    total = state["recyclable"] + state["organic"]
    return f"""
    <div class="chips">
        <div class="chip"><b>{total}</b><span>Scanned</span></div>
        <div class="chip" style="--accent:#3b82f6"><b>{state['recyclable']}</b><span>♻️ Recyclable</span></div>
        <div class="chip" style="--accent:#10b981"><b>{state['organic']}</b><span>🍎 Organic</span></div>
    </div>"""


def classify(image, state):
    if image is None:
        return (placeholder_card("⚠️ No image received. Upload or capture one first."),
                state["hist"], stats_html(state), state)
    try:
        t0 = time.time()
        p = float(pred_model.predict(preprocess(image), verbose=0)[0][0])
        secs = time.time() - t0

        key = "recyclable" if p > 0.5 else "organic"
        conf = p if key == "recyclable" else 1 - p

        state[key] += 1
        thumb = Image.fromarray(np.asarray(image).astype("uint8")[..., :3])
        thumb.thumbnail((220, 220))
        state["hist"].insert(0, (thumb, f"{CATEGORIES[key]['emoji']} {key.title()} · {conf:.0%}"))
        state["hist"] = state["hist"][:8]

        return result_card(key, conf, secs), state["hist"], stats_html(state), state
    except Exception as e:
        return placeholder_card(f"❌ Prediction error: {e}"), state["hist"], stats_html(state), state


def auto_classify(image, auto, state):
    if not auto or image is None:
        return gr.update(), gr.update(), gr.update(), state
    return classify(image, state)


def clear_all(state):
    return None, placeholder_card()


def new_state():
    return {"recyclable": 0, "organic": 0, "hist": []}


custom_css = """
.gradio-container {
    max-width: 1100px !important; margin: auto !important;
    background: linear-gradient(160deg, #ecfdf5 0%, #eff6ff 50%, #f5f3ff 100%) !important;
}
footer {display:none !important;}

/* ---------- Hero ---------- */
.hero {text-align:center; padding:38px 16px; border-radius:24px; color:#fff; position:relative; overflow:hidden;
       background:linear-gradient(135deg,#059669,#0ea5e9,#6366f1); background-size:200% 200%;
       animation:gradientMove 8s ease infinite; box-shadow:0 18px 40px -12px rgba(14,165,233,.55);}
.hero h1 {margin:0; font-size:2.6rem; font-weight:800; color:#fff; letter-spacing:-.5px;}
.hero p {margin:10px auto 0; font-size:1.05rem; opacity:.95; max-width:620px;}
.hero .logo {font-size:3.4rem; display:inline-block; animation:floaty 3s ease-in-out infinite;}
@keyframes gradientMove {0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}
@keyframes floaty {0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)}}

/* ---------- How it works ---------- */
.steps {display:flex; gap:14px; margin:18px 0 6px;}
.step {flex:1; text-align:center; padding:16px 10px; border-radius:16px; background:#fff; color:#0f172a;
       box-shadow:0 6px 18px -8px rgba(15,23,42,.18); border:1px solid #e2e8f0;}
.step .n {display:inline-block; width:28px; height:28px; line-height:28px; border-radius:50%;
          background:linear-gradient(135deg,#10b981,#3b82f6); color:#fff; font-weight:700; margin-bottom:6px;}
.step b {display:block; font-size:.98rem;}
.step span {font-size:.82rem; color:#64748b;}

/* ---------- Result card ---------- */
.card {border-radius:20px; padding:24px; background:#fff; color:#0f172a;
       border:1px solid #e2e8f0; border-top:6px solid var(--accent,#9ca3af);
       box-shadow:0 14px 34px -14px rgba(15,23,42,.28); animation:popIn .45s ease;}
.card.empty {text-align:center; padding:56px 20px; color:#64748b; --accent:#cbd5e1;}
.card .row {display:flex; align-items:center; gap:16px;}
.card .big {font-size:3.4rem; line-height:1;}
.card h2 {margin:6px 0 0; font-size:1.6rem; font-weight:800; color:#0f172a;}
.card h4 {margin:18px 0 6px; color:#0f172a;}
.card ul {margin:0; padding-left:20px; line-height:1.8; color:#334155;}
.badge {display:inline-block; padding:4px 14px; border-radius:999px; background:var(--accent);
        color:#fff; font-weight:700; font-size:.8rem; letter-spacing:.04em;}
.meta {display:flex; justify-content:space-between; margin:18px 0 8px; font-size:.95rem; color:#334155;}
.bar {height:14px; border-radius:999px; background:#e2e8f0; overflow:hidden;}
.bar > div {height:100%; background:linear-gradient(90deg,var(--accent),#ffffff55); border-radius:999px;
            transition:width .8s ease;}
.warn {margin-top:14px; padding:10px 14px; border-radius:12px; background:#fffbeb;
       border:1px solid #fcd34d; color:#92400e; font-size:.92rem;}
.foot {margin-top:16px; font-size:.78rem; color:#94a3b8; text-align:right;}
@keyframes popIn {from{opacity:0; transform:translateY(12px) scale(.98)} to{opacity:1; transform:none}}

/* ---------- Stats chips ---------- */
.chips {display:flex; gap:14px;}
.chip {flex:1; text-align:center; padding:16px; border-radius:16px; background:#fff; color:#0f172a;
       border:1px solid #e2e8f0; border-bottom:5px solid var(--accent,#94a3b8);
       box-shadow:0 8px 20px -10px rgba(15,23,42,.2); transition:transform .2s ease;}
.chip:hover {transform:translateY(-3px);}
.chip b {display:block; font-size:1.9rem;}
.chip span {font-size:.85rem; color:#64748b;}

/* ---------- Buttons ---------- */
.action-btn {background:linear-gradient(135deg,#10b981,#3b82f6) !important; color:#fff !important;
             font-weight:700 !important; font-size:1.02rem !important; border:none !important;
             border-radius:14px !important; transition:transform .2s ease, box-shadow .2s ease !important;}
.action-btn:hover {transform:translateY(-3px); box-shadow:0 14px 24px -8px rgba(59,130,246,.6) !important;}

/* ---------- Upload box ---------- */
.upload-box {
    border: 2.5px dashed #34d399 !important;
    border-radius: 22px !important;
    background: linear-gradient(180deg,#ffffff 0%,#f0fdf4 100%) !important;
    box-shadow: 0 14px 30px -16px rgba(16,185,129,.55) !important;
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease !important;
}
.upload-box:hover {
    transform: translateY(-4px);
    border-color: #3b82f6 !important;
    box-shadow: 0 22px 36px -14px rgba(59,130,246,.5) !important;
}
.upload-box * {font-weight: 600;}
.upload-box label, .upload-box .label-wrap {font-size: 1rem !important;}

/* ---------- Empty result card ---------- */
.card.empty {
    background: linear-gradient(160deg,#ffffff 0%,#eff6ff 100%);
    border-top: 6px solid #3b82f6;
    padding: 44px 24px;
}
.card.empty .big {font-size: 4.2rem; animation: floaty 3s ease-in-out infinite;}
.card.empty h3 {margin: 10px 0 4px; font-size: 1.3rem; font-weight: 800; color:#0f172a;}
.card.empty p {margin: 0 0 18px; color:#64748b;}
.hints {display:flex; flex-wrap:wrap; gap:8px; justify-content:center;}
.hints span {padding:6px 12px; border-radius:999px; background:#ecfdf5; color:#047857;
             font-size:.82rem; font-weight:600; border:1px solid #a7f3d0;}
.hints span.b {background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe;}
"""

theme = gr.themes.Soft(
    primary_hue="emerald", secondary_hue="sky", neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("Poppins"), "system-ui", "sans-serif"],
)



with gr.Blocks(theme=theme, css=custom_css, title="BinGENIUS Smart Classifier") as app:
    state = gr.State(new_state())
    gr.HTML("""
     <div class="hero">
        <div class="logo">♻️</div>
        <h1>BinGENIUS Smart Classifier</h1>
        <p>AI-powered waste sorting. Upload, paste or snap a photo and instantly find the right bin.</p>
    </div>
    <div class="steps">
        <div class="step"><div class="n">1</div><b>Add a photo</b><span>Upload, webcam or paste</span></div>
        <div class="step"><div class="n">2</div><b>AI analyses it</b><span>VGG16 CNN model</span></div>
        <div class="step"><div class="n">3</div><b>Get the right bin</b><span>With disposal tips</span></div>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            input_img = gr.Image(sources=["upload", "webcam", "clipboard"], type="numpy",
                                 label="📸 Drop your waste photo here", height=380,
                                 elem_classes="upload-box")
            auto = gr.Checkbox(value=True, label="Auto-classify when an image is added")
            with gr.Row():
                submit_btn = gr.Button("Identify Target Bin 🔍", elem_classes="action-btn", scale=3)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
        with gr.Column(scale=1):
            result = gr.HTML(placeholder_card())

    gr.Markdown("### 📊 Session stats")
    stats = gr.HTML(stats_html(new_state()))
    history = gr.Gallery(label="Recent scans", columns=4, height=210, object_fit="cover", interactive=False)

    gr.Markdown("<center><small>Binary CNN classifier (recyclable vs organic). Results are a guide, "
                "so check your local waste rules for edge cases.</small></center>")

    outs = [result, history, stats, state]
    submit_btn.click(classify, [input_img, state], outs)
    input_img.change(auto_classify, [input_img, auto, state], outs)
    clear_btn.click(clear_all, state, [input_img, result])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(
        server_name="0.0.0.0",
        server_port=port
    )