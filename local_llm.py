from llama_cpp import Llama

MODEL_PATH = "../models/llama-2-7b.Q4_K_M.gguf"

llm = Llama(model_path=MODEL_PATH, verbose=False)

def llama2(prompt, max_tokens=32):
    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            echo=False
        )
        
        answer = output['choices'][0]['text']
        
        answer_lines = answer.split("\n", 1)
        
        if len(answer_lines) > 1:
            return answer_lines[1].strip() 
        else:
            return answer.strip()
    
    except Exception as e:
        print(f"Llama2 Model Error: {e}")
        return ""

def test_model():
    test_question = "Is Rome the capital of Italy?"
    result = llama2(test_question)
    print(f"Test Question: {test_question}")
    print(f"Model Response: {result}")

if __name__ == "__main__":
    test_model()