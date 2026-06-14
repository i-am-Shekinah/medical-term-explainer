import gradio as gr
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

prompt_template_str = """
You are a friendly, compassionate medical educator skilled in translating complex 
clinical jargon into simple, plain language for patients and their families.

Your task is to explain the following medical term: {medical_term}

Please structure your response as follows:
1. **The "In a Nutshell" Definition:** A one-sentence, non-technical summary.
2. **Plain Language Explanation:** Break down what it is, using a simple analogy if possible.
3. **Common Context:** Briefly mention why someone might hear this term (e.g., a specific test, symptom, or condition).
4. **When to Ask a Doctor:** Suggest 1-2 specific questions the user should ask their healthcare provider to better understand their situation.

Constraints:
- Avoid Latin-based medical jargon unless immediately defined.
- Use a warm, supportive, and objective tone.
- Add a disclaimer at the end stating: "This information is for educational purposes and does not replace professional medical advice, diagnosis, or treatment."
"""

prompt_template = PromptTemplate.from_template(prompt_template_str)

model = init_chat_model('gemini-3.5-flash', model_provider='google_genai')


def generate_explanation(medical_term: str):
    prompt = prompt_template.format(medical_term=medical_term)

    response = model.invoke(prompt)

    return response.text


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown('# 🩺 Medical Term Explainer')
    gr.Markdown('### Understand medical jargon in simple, human language')

    status = gr.Markdown('🟢 Ready')

    with gr.Row():
        term_input = gr.Textbox(
            label='Enter a medical term',
            placeholder='e.g. Myocardial Infarction',
        )

    submit_btn = gr.Button('Explain', variant='primary')

    output = gr.Markdown()

    def wrapped_generate(term):
        yield '⏳ Thinking... analyzing medical term', ''
        result = generate_explanation(term)
        yield "✅ Done", result

    submit_btn.click(
        fn=wrapped_generate,
        inputs=term_input,
        outputs=[status, output]
    )

if __name__ == '__main__':
    demo.launch()

