# src/ui.py
import gradio as gr
from PIL import Image
from src.utils import IMAGE_SIZE
from src.comic_logic import generate_comic

def build_interface():
    """Build and return a friendlier, more colorful Gradio UI."""
    custom_css = """
    body {
        background-color: #f0f8ff !important;
    }
    .gradio-container {
        background-color: #ffffff !important;
        border-radius: 15px !important;
        padding: 20px !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important;
    }
    #panel1, #panel2, #panel3, #panel4 {
        border: 3px solid #4a90e2 !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    .gradio-title {
        color: #4a90e2 !important;
        font-family: 'Arial', sans-serif !important;
        font-weight: bold !important;
    }
    .gradio-markdown {
        color: #333333 !important;
        font-family: 'Arial', sans-serif !important;
    }
    /* Make the user sketch smaller */
    .small-upload img, .small-upload .gr-image, .small-upload .gradio-image {
        max-width: 200px !important;
        height: auto !important;
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css) as interface:
        with gr.Row(variant="panel"):
            with gr.Column(scale=8):
                gr.Markdown("# 🎨 ComicCrafter AI", elem_classes="gradio-title")
                gr.Markdown("### Generate a 4-panel comic strip from your prompt", elem_classes="gradio-markdown")
        
        with gr.Row():
            with gr.Column(scale=3):
                with gr.Group():
                    gr.Markdown("### Story Prompt", elem_classes="gradio-markdown")
                    prompt_input = gr.Textbox(
                        label="Your story idea",
                        placeholder="Enter your prompt, e.g. 'A superhero adventure'",
                        lines=4
                    )
                    gr.Markdown("### Upload a Sketch for ControlNet", elem_classes="gradio-markdown")
                    user_sketch = gr.Image(
                        label="Your sketch or reference",
                        type="pil",
                        image_mode="RGB",
                        value=Image.new("RGB", IMAGE_SIZE, color="white"),
                        elem_classes="small-upload"
                    )
                    gr.Markdown("### Character Library (JSON)", elem_classes="gradio-markdown")
                    char_lib_input = gr.Textbox(
                        label="Recurring Characters",
                        placeholder='{"Hero": "a brave young man with blue eyes"}',
                        lines=3
                    )
                    gr.Markdown("### Generation Settings", elem_classes="gradio-markdown")
                    num_steps_slider = gr.Slider(10, 50, step=1, value=25, label="Inference Steps")
                    guidance_slider = gr.Slider(1.0, 15.0, step=0.5, value=7.5, label="Guidance Scale")
                    controlnet_slider = gr.Slider(0.1, 2.0, step=0.1, value=0.8, label="ControlNet Conditioning Scale")

                    gr.Markdown("#### Examples:", elem_classes="gradio-markdown")
                    with gr.Row():
                        example1 = gr.Button("🦸‍♂️ Superhero Adventure")
                        example2 = gr.Button("🧙 Magical World")
                    with gr.Row():
                        example3 = gr.Button("🚀 Space Journey")
                        example4 = gr.Button("🐉 Mythical Tale")

                    with gr.Row():
                        clear_btn = gr.Button("Clear", variant="secondary")
                        submit_btn = gr.Button("Generate ✨", variant="primary")
                
                with gr.Group(visible=True):
                    gr.Markdown("- **© 2025 ComicCrafter AI**", elem_classes="gradio-markdown")
                    gr.Markdown("- **Developed By: Himanshu Kumar**", elem_classes="gradio-markdown")

            with gr.Column(scale=7):
                gr.Markdown("## 🖼️ Your Comic Strip", elem_classes="gradio-title")
                output_status = gr.Markdown("Your comic will appear here after generating.", elem_classes="gradio-markdown")
                with gr.Row():
                    image1 = gr.Image(label="Introduction", elem_id="panel1")
                    image2 = gr.Image(label="Development", elem_id="panel2")
                with gr.Row():
                    image3 = gr.Image(label="Climax", elem_id="panel3")
                    image4 = gr.Image(label="Conclusion", elem_id="panel4")
        
        # Example prompt callbacks
        example1.click(fn=lambda: "A young boy who gains the power of flight and protects his city", outputs=prompt_input)
        example2.click(fn=lambda: "A wizard who enters a mysterious forest in search of a lost book", outputs=prompt_input)
        example3.click(fn=lambda: "Space travelers stranded on an unknown planet trying to find their way home", outputs=prompt_input)
        example4.click(fn=lambda: "A story of friendship between a dragon and a princess who together save the kingdom", outputs=prompt_input)
        clear_btn.click(fn=lambda: "", outputs=prompt_input)

        def show_generating_message():
            return "⏳ Generating your comic... Please wait (60-120 seconds)..."

        def show_completed_message():
            return "✅ Your comic is ready!"

        submit_btn.click(
            fn=show_generating_message,
            outputs=output_status
        ).then(
            fn=generate_comic,
            inputs=[prompt_input, user_sketch, num_steps_slider, guidance_slider, controlnet_slider, char_lib_input],
            outputs=[image1, image2, image3, image4]
        ).then(
            fn=show_completed_message,
            outputs=output_status
        )

    interface.queue()
    return interface