from llama_cpp import Llama
model_path = "../models/llama-2-7b.Q4_K_M.gguf"

question = "Is Rome the capital of Italy? "
llm = Llama(model_path=model_path, verbose=False)
print("Asking the question \"%s\" to %s (wait, it can take some time...)" % (question, model_path))
output = llm(
    question,  # Prompt
    max_tokens=32,  # Increase max_tokens to get longer answers
    echo=False  # Do not echo the prompt back in the output
)

# Print full output for debugging
#print("Full Output: ", output)

# Extract both short and long answers by splitting at the first newline
answer = output['choices'][0]['text']
answer_lines = answer.split("\n", 1)  # Split at the first newline

# Extract both parts: before and after the first newline
if len(answer_lines) > 1:
    answer1 = answer_lines[0].strip()  # Short answer (before newline)
    answer2 = answer_lines[1].strip()  # Long answer (after newline)
else:
    answer1 = answer.strip()  # If no newline, just return the whole answer as short answer
    answer2 = ""  # Set long answer as empty if there's no newline

# Print both parts
print("Answer_1: ", answer1)
print("Answer_2: ", answer2)