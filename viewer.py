import os
import gradio as gr
from PIL import Image

# Base directory
BASE_DIR = "data/output"

# Helper constants
VIEW_MODES = ["abs", "muscle", "overlap"]
CATEGORY_MAP = {"Inhalation": "in", "Exhalation": "ex"}

def get_patient_ids():
    """Retrieve patient IDs from the base directory."""
    patient_id_list = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    patient_id_list.sort()
    return patient_id_list


# New function: get_view_modes
def get_view_modes(patient_id):
    """Return existing view‑mode sub‑directories (abs/muscle/overlap) for the patient."""
    patient_dir = os.path.join(BASE_DIR, patient_id)
    return [
        d for d in VIEW_MODES
        if os.path.isdir(os.path.join(patient_dir, d))
    ]

def get_categories(patient_id, view_mode):
    """Return available categories (in / ex) for the selected view mode."""
    if not patient_id or not view_mode:
        return []
    mode_path = os.path.join(BASE_DIR, patient_id, view_mode)
    raw = [
        d for d in os.listdir(mode_path)
        if os.path.isdir(os.path.join(mode_path, d))
    ] if os.path.exists(mode_path) else []
    # 'in' → Inhalation,  'ex' → Exhalation
    return ["Inhalation" if d == "in" else "Exhalation" for d in sorted(raw)]

def get_series_numbers(patient_id, view_mode, category):
    """Return series numbers inside .../patient/view_mode/category."""
    if not (patient_id and view_mode and category):
        return []
    cat_dir = CATEGORY_MAP.get(category, "")
    series_root = os.path.join(BASE_DIR, patient_id, view_mode, cat_dir)
    if not os.path.exists(series_root):
        return []
    return sorted([
        s for s in os.listdir(series_root)
        if os.path.isdir(os.path.join(series_root, s))
    ])

def get_images(patient_id, view_mode, category, series_number):
    """Return image paths for the fully‑specified 4‑depth directory."""
    if not all([patient_id, view_mode, category, series_number]):
        return []
    cat_dir = CATEGORY_MAP.get(category, "")
    img_root = os.path.join(BASE_DIR, patient_id, view_mode, cat_dir, series_number)
    if not os.path.exists(img_root):
        return []
    return [
        os.path.join(img_root, f)
        for f in os.listdir(img_root)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

def display_images(patient_id, view_mode, category, series_number):
    paths = sorted(get_images(patient_id, view_mode, category, series_number))
    return [(Image.open(p), os.path.basename(p).split(".")[0]) for p in paths]

def update_view_modes(patient_id):
    choices = get_view_modes(patient_id)
    return gr.update(choices=choices, value=choices[0] if choices else None)

def update_categories(patient_id, view_mode):
    choices = get_categories(patient_id, view_mode)
    return gr.update(choices=choices, value=choices[0] if choices else None)

def update_series_numbers(patient_id, view_mode, category):
    choices = get_series_numbers(patient_id, view_mode, category)
    return gr.update(choices=choices, value=choices[0] if choices else None)

def update_gallery(patient_id, view_mode, category, series_number):
    return display_images(patient_id, view_mode, category, series_number)

css_value = """

"""

with gr.Blocks(title="Rib Muscle Segmentation Viewer", css=css_value) as demo:
    gr.Markdown("## Rib Muscle Segmentation Viewer")
    gr.Markdown("추출된 결과를 확인할 수 있는 웹 뷰어입니다.")

    with gr.Row():
        patient_id_dropdown  = gr.Dropdown(label="환자 ID", choices=get_patient_ids(), interactive=True)
        view_mode_dropdown   = gr.Dropdown(label="보기 모드", choices=[], interactive=True)
        category_dropdown    = gr.Dropdown(label="카테고리", choices=[], interactive=True)
        series_dropdown      = gr.Dropdown(label="시리즈 번호", choices=[], interactive=True)

    image_gallery = gr.Gallery(label="결과 (Results)", rows=5, columns=5, show_fullscreen_button=True)

    # depth‑wise cascading updates
    patient_id_dropdown.change(
        fn=update_view_modes,
        inputs=patient_id_dropdown,
        outputs=view_mode_dropdown
    )

    view_mode_dropdown.change(
        fn=update_categories,
        inputs=[patient_id_dropdown, view_mode_dropdown],
        outputs=category_dropdown
    )

    category_dropdown.change(
        fn=update_series_numbers,
        inputs=[patient_id_dropdown, view_mode_dropdown, category_dropdown],
        outputs=series_dropdown
    )

    # gallery refresh whenever any selector changes
    for dd in (patient_id_dropdown, view_mode_dropdown, category_dropdown, series_dropdown):
        dd.change(
            fn=update_gallery,
            inputs=[patient_id_dropdown, view_mode_dropdown, category_dropdown, series_dropdown],
            outputs=image_gallery
        )

    demo.load(
        fn=update_gallery,
        inputs=[patient_id_dropdown, view_mode_dropdown, category_dropdown, series_dropdown],
        outputs=image_gallery
    )

demo.launch(share=False)