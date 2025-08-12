import gradio as gr
from transformers import pipeline
import os

# Initialize the text generation pipeline with a free Hugging Face model
try:
    generator = pipeline(
        "text2text-generation", 
        model="google/flan-t5-large",
        max_length=512,
        temperature=0.7
    )
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to a smaller model if large one fails
    generator = pipeline(
        "text2text-generation", 
        model="google/flan-t5-base",
        max_length=512,
        temperature=0.7
    )

def generate_story(problem, background, affected_people, emotions, call_to_action):
    """Generate an emotional impact story based on user inputs"""
    
    if not all([problem, background, affected_people, emotions, call_to_action]):
        return "Please fill in all fields to generate your story."
    
    # Create a structured prompt for story generation
    prompt = f"""Write an inspiring and emotional story about a community issue. 

Problem: {problem}
Background: {background}
Who is affected: {affected_people}
Emotional impact: {emotions}
Call to action: {call_to_action}

Create a compelling 200-300 word story that:
- Opens with a human connection
- Explains the problem clearly
- Shows the emotional impact on people
- Ends with an inspiring call to action
- Uses vivid, emotional language
- Motivates readers to care and act

Story:"""

    try:
        # Generate the story
        result = generator(
            prompt, 
            max_length=400,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        
        story = result[0]['generated_text']
        
        # Clean up the story if needed
        if story.startswith(prompt):
            story = story.replace(prompt, "").strip()
        
        # Add a nice header
        formatted_story = f"🌟 **Your Social Good Story** 🌟\n\n{story}\n\n---\n\n💙 Share this story to make a difference in your community!"
        
        return formatted_story
        
    except Exception as e:
        return f"Sorry, there was an error generating your story. Please try again. Error: {str(e)}"

# Create the Gradio interface
def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Social Good Storytelling Chatbot",
        css="""
        .gradio-container {
            max-width: 800px !important;
            margin: auto !important;
        }
        .title {
            text-align: center;
            margin-bottom: 30px;
        }
        .story-output {
            min-height: 300px;
        }
        """
    ) as app:
        
        # Header
        gr.HTML("""
            <div class="title">
                <h1>🌟 Social Good Storytelling Chatbot 🌟</h1>
                <p><em>Transform community needs into compelling stories that inspire action</em></p>
            </div>
        """)
        
        gr.Markdown("### Tell us about a community issue, and we'll help you create a powerful story to drive change!")
        
        with gr.Row():
            with gr.Column():
                # Input fields
                problem = gr.Textbox(
                    label="1. What is the main problem or need in your community?",
                    placeholder="e.g., Lack of safe spaces for children to play, food insecurity, inadequate public transportation...",
                    lines=2,
                    max_lines=3
                )
                
                background = gr.Textbox(
                    label="2. What's the background/context of this issue?",
                    placeholder="e.g., Due to urban development, the only park was converted to a shopping mall, leaving 500+ families without...",
                    lines=2,
                    max_lines=3
                )
                
                affected_people = gr.Textbox(
                    label="3. Who is most affected by this problem?",
                    placeholder="e.g., Working mothers with young children, elderly residents who rely on public transport, local small business owners...",
                    lines=2,
                    max_lines=3
                )
                
                emotions = gr.Textbox(
                    label="4. What emotions or struggles do people face because of this?",
                    placeholder="e.g., Parents feel anxious about their children's safety, families feel isolated, students miss opportunities...",
                    lines=2,
                    max_lines=3
                )
                
                call_to_action = gr.Textbox(
                    label="5. What action do you want people to take?",
                    placeholder="e.g., Sign the petition for a new community center, volunteer at the local food bank, contact city council...",
                    lines=2,
                    max_lines=3
                )
                
                # Generate button
                generate_btn = gr.Button(
                    "✨ Generate My Social Good Story ✨",
                    variant="primary",
                    size="lg"
                )
        
        # Output area
        gr.Markdown("### Your Generated Story:")
        story_output = gr.Textbox(
            label="",
            placeholder="Your inspiring story will appear here...",
            lines=15,
            max_lines=20,
            elem_classes=["story-output"],
            show_copy_button=True
        )
        
        # Example section
        gr.Markdown("""
        ### 💡 Need inspiration? Here are some example community issues:
        - **Education**: Lack of after-school programs, insufficient school supplies, digital divide
        - **Environment**: Plastic waste, lack of recycling facilities, air pollution
        - **Health**: Mental health resources, access to healthy food, medical care for seniors
        - **Housing**: Affordable housing shortage, homelessness, unsafe living conditions
        - **Transportation**: Poor public transit, unsafe walking conditions, limited accessibility
        """)
        
        # Connect the generate button to the function
        generate_btn.click(
            fn=generate_story,
            inputs=[problem, background, affected_people, emotions, call_to_action],
            outputs=story_output
        )
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #f0f0f0; border-radius: 10px;">
                <p><strong>🎯 How to use your story:</strong></p>
                <p>• Share on social media with relevant hashtags<br>
                • Include in grant applications or fundraising materials<br>
                • Present to local government or community leaders<br>
                • Use in newsletters, blogs, or community presentations</p>
                <p><em>Made with ❤️ for social good initiatives</em></p>
            </div>
        """)
    
    return app

# Launch the app
if __name__ == "__main__":
    app = create_interface()
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
