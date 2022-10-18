import os

import gradio as gr
import spacy
from spacy import displacy

from articles import *

os.system('python -m spacy download en_core_web_trf en_core_web_sm')

def ner(text, model='en_core_web_trf'):
    ner_model = spacy.load(model)
    doc = ner_model(text)
    dep_tree = displacy.render(doc, style="dep", page=True)
    dep_tree = (
        ""
        + dep_tree
        + ""
    )
    doc = ner_model(text)
    meta_data = {
        "char_count": len(text),
        "token_count": len(doc),
    }
    pos_tokens = []
    for token in doc:
        pos_tokens.extend([(token.text, token.pos_), (" ", None)])
    return pos_tokens, meta_data, dep_tree

demo = gr.Blocks()  # Create a gradio block

with demo:
    gr.Markdown("# Token Classification with Spacy")
    gr.Markdown(
        "### Enter the text you want to Classify or choose from examples below")
    with gr.Tabs():
        with gr.TabItem("Examples"):  # If the user wants to use the examples
            with gr.Column():
                rad = gr.components.Radio(
                    ['Article 1', 'Article 2', 'Article 3'], label='Select article and wait till it appears!')  # Radio button to select the article
                # Textbox to show the article
                text1 = gr.Textbox(label='Example')
                rad2 = gr.components.Radio(
                    ['Spacy English Transformer', 'Spacy English Small'], label='Select Model for Token Classification')  # Radio button to select the model
            submit1 = gr.Button('Submit')
        with gr.TabItem("Do it yourself!"):  # If the user wants to enter their own text
            with gr.Column():
                text2 = gr.components.Textbox(
                    label='Enter your own text here!')
                rad3 = gr.components.Radio(
                    ['Spacy English Transformer', 'Spacy English Small'], label='Select Model for Token Classification')  # Radio button to select the model
            submit2 = gr.Button('Submit')

        def action1(choice):  # Function to show the article when the user selects the article
            if choice == 'Article 1':
                return ARTICLE_1
            elif choice == 'Article 2':
                return ARTICLE_2
            elif choice == 'Article 3':
                return ARTICLE_3

        def models(model_name):  # Function to select the model
            if model_name == 'Spacy English Transformer':
                return 'en_core_web_trf'
            elif model_name == 'Spacy English Small':
                return 'en_core_web_sm'
            elif model_name is None:
                return 'en_core_web_trf'

        # Change the article when the user selects the article
        rad.change(action1, rad, text1)

        # Output for the Highlighted text
        op = gr.components.HighlightedText(label='Classified Token')
        op2 = gr.components.JSON(label='Meta')  # Output for the Meta data
        # Output for the Dependency Tree
        op3 = gr.components.HTML(label='Dependency Tree')
        gr.Markdown(
            "### Made with ❤️ by Arsh using TrueFoundry's Gradio Deployment")
        gr.Markdown(
            "### [Github Repo](https://github.com/d4rk-lucif3r/Token-Classification-with-Spacy)")
        gr.Markdown(
            '### [Blog]()')

        def fn(text, model):  # Main function
            model = models(model)
            result = ner(text, model)
            return result

        submit1.click(fn=fn, inputs=[text1, rad2], outputs=[
                      op, op2, op3])  # Submit button for the examples
        # Submit button for the user input
        submit2.click(fn=fn, inputs=[text2, rad3], outputs=[op, op2, op3])

demo.queue()
demo.launch(server_port=8080, server_name='0.0.0.0') # Launch the gradio block
