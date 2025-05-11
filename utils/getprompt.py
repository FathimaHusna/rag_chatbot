from langchain_core.prompts import PromptTemplate

def get_prompt():
    """
    Create a prompt template for fortune telling.
    """
    # Define the template
    template = """
    You are a fortune teller. These Human will ask you a questions about their life. 
    Use following piece of context to answer the question. 
    If you don't know the answer, just say you don't know. 
    Keep the answer within 2 sentences and concise.

    Context: {context}
    Question: {question}
    Answer: 

    """

    prompt = PromptTemplate(
    template=template, 
    input_variables=["context", "question"]
    )

    return prompt