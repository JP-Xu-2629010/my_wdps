import spacy
import requests
from sentence_transformers import SentenceTransformer, util
import time
from local_llm import llama2


nlp = spacy.load("en_core_web_sm")

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
LLAMA_API_URL = "http://localhost:11434/api/chat" 

def llama3(prompt):
    payload = {
        "model": "llama3.1",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = requests.post(LLAMA_API_URL, headers={"Content-Type": "application/json"}, json=payload)
    if response.status_code == 200:
        return response.json().get("message", {}).get("content")
    else:
        print(f"Llama API Error: {response.status_code}")
        return ""

def search_wikipedia(entity):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": entity,
        "format": "json",
        "srlimit": 20
    }
    response = requests.get(WIKIPEDIA_API_URL, params=params)
    if response.status_code != 200:
        print(f"Error querying Wikipedia Search API: {response.status_code}, Response: {response.text}")
        return []

    data = response.json()

    # Validate response structure
    if "query" not in data or "search" not in data["query"]:
        print(f"No search results for entity: {entity}")
        return []

    results = []
    for result in data["query"]["search"]:
        title = result["title"]
        snippet = result.get("snippet", "").replace("<span class='searchmatch'>", "").replace("</span>", "")
        results.append((title, snippet))
    return results

# Fetch Wikipedia Page Summaries
def fetch_wikipedia_summary(title):
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "format": "json"
    }
    response = requests.get(WIKIPEDIA_API_URL, params=params)
    if response.status_code != 200:
        print(f"Error fetching Wikipedia summary: {response.status_code}")
        return None

    data = response.json()
    pages = data["query"]["pages"]
    for page_id, page_data in pages.items():
        if "extract" in page_data:
            return page_data["extract"]
    return None

def get_embedding(text):
    return embedding_model.encode(text, convert_to_tensor=True)

def extract_entities(text):
    doc = nlp(text)
    return [ent.text.strip() for ent in doc.ents]

# Excludes numbers
def is_valid_entity(entity):
    return not entity.isdigit()

# Disambiguate entity using embeddings and context
def disambiguate_entity(entity, context):

    time.sleep(0.1)

    context_embedding = get_embedding(f"{entity} {context}")
    candidates = search_wikipedia(entity)

    if not candidates:
        return entity, None  # Return the entity with no match

    ranked_candidates = []

    for title, snippet in candidates:
        summary = fetch_wikipedia_summary(title) or snippet
        candidate_text = f"{title} {summary}"

        candidate_embedding = get_embedding(candidate_text)

        similarity_score = util.pytorch_cos_sim(context_embedding, candidate_embedding).item()

        wikipedia_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        ranked_candidates.append((similarity_score, title, wikipedia_url))
    
    ranked_candidates.sort(reverse=True, key=lambda x: x[0])

    if ranked_candidates:
        _, label, best_url = ranked_candidates[0]
        return entity, best_url
    return entity, None

def process_question(question):

    #llama_reply = llama3(question)
    #print(f"Llama3 Answer: {llama_reply}")
    llama_reply = llama2(question)
    print(f"Llama2 Answer: {llama_reply}")

    context = f"{question} {llama_reply}"

   
    all_entities = extract_entities(context)
    print(f"Extracted entities: {all_entities}")

    results = []
    for entity in all_entities:
        entity_name, best_url = disambiguate_entity(entity, context)
        if best_url:
            # Output: Entity → Wikipedia link
            results.append(f"{entity_name} ⇒ {best_url}")

    if results:
        print("\n".join(results))
    else:
        print("No suitable matches found.")

def main():
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        question = input("Enter a question: ").strip()
        if question.lower() == "exit":
            break

        process_question(question)
        print("\n------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
