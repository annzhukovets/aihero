from openai import OpenAI

def llm(prompt, model='gpt-4o-mini'):
    openai_client = OpenAI()
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    response = openai_client.responses.create(
        model='gpt-4o-mini',
        input=messages
    )

    return response.output_text

def intelligent_chunking(text):
    prompt_template = """
        Split the provided document into logical sections
        that make sense for a Q&A system.
        
        Each section should be self-contained and cover
        a specific topic or concept.
        
        <DOCUMENT>
        {document}
        </DOCUMENT>
        
        Use this format:
        
        ## Section Name
        
        Section content with all relevant details
        
        ---
        
        ## Another Section Name
        
        Another section content
        
        ---
        """.strip()
    
    prompt = prompt_template.format(document=text)
    response = llm(prompt)
    sections = response.split('---')
    sections = [s.strip() for s in sections if s.strip()]
    return sections