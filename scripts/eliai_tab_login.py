import modules.scripts as scripts
import gradio as gr
import os

import eliai_auth

from modules import script_callbacks, shared

def read_user_token():
    with open(f'{shared.cmd_opts.eliai_lora_dir}\\token.txt') as f:
        result = f.read()
    return result

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        title = gr.Markdown(value="")
        with gr.Row():
            with gr.Column():
                token_input = gr.Textbox(
                    placeholder="Nhập token của bạn vào đây",
                    value='',
                    label="Token",
                    interactive=True
                )
            with gr.Column():
                login_btn = gr.Button(
                    value="Đăng nhập",
                    scale=2,
                    variant="primary"
                )
                login_btn.click(login, inputs=[token_input], outputs=title)
                with gr.Row():
                    logout_btn = gr.Button(
                        value="Đăng xuất",
                        scale=1,
                        variant="secondary"
                    )
                    logout_btn.click(logout, inputs=[token_input], outputs=title)
                    login_check_btn = gr.Button(
                        value="Check thông tin đăng nhập",
                        size='sm',
                        variant="secondary"
                    )
                    login_check_btn.click(fn=login_check, outputs=title)
            # TODO: add more UI components (cf. https://gradio.app/docs/#components)
        return [(ui_component, "EliAI", "extension_template_tab")]
     
def login(token_input):
    result = eliai_auth.login(token_input)
    if result != -1:
        return gr.update(value="**Đã đăng nhập thành công, bạn có thể sử dụng bình thường**")
    else:
        return gr.update(value="**Đăng nhập thất bại, vui lòng kiểm tra lại token**")

def logout(token_input):
    result = eliai_auth.logout(token_input)
    if result != -1:
        return gr.update(value="**Vui lòng đăng nhập**")
    else:
        return gr.update(value="**Đăng xuất thất bại, vui lòng thử lại**")

def login_check():
    token = read_user_token()
    if not token:
        return gr.update(value="**Vui lòng đăng nhập để sử dụng lora EliAI**")
    else:
        return gr.update(value="**Đã đăng nhập thành công,bạn có thể sử dụng bình thường**")

script_callbacks.on_ui_tabs(on_ui_tabs)