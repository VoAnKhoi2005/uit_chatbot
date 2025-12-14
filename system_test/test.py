import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests
import time
from tqdm import tqdm
import os

EXCEL_FILE = r"E:\Github\uit_chatbot\system_test\raw_excel\train.xlsx"
API_URL = "http://localhost:10000/chat"
OUTPUT_CSV = r"E:\Github\uit_chatbot\system_test\test_results.csv"
TEST_AMOUNT = 200

model = SentenceTransformer("all-MiniLM-L6-v2")

def keyword_exact_match(prediction, keywords):
    prediction_norm = prediction.lower()
    return int(all(kw.lower() in prediction_norm for kw in keywords))

def keyword_f1(prediction, keywords):
    prediction_tokens = set(prediction.lower().split())
    keywords_tokens = set(kw.lower() for kw in keywords)
    common = prediction_tokens & keywords_tokens
    if not common:
        return 0.0
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(keywords_tokens)
    return 2 * precision * recall / (precision + recall)

def bert_score(prediction, reference):
    embeddings1 = model.encode([prediction], convert_to_tensor=True)
    embeddings2 = model.encode([reference], convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_scores.item()

def call_chatbot_api(question):
    try:
        payload = {"question": question, "conversation_history": None}
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("answer", ""), data.get("question_type", "")
    except Exception as e:
        print(f"API call failed for question: {question}")
        print(f"Error: {e}")
        return "", "ERROR"

def main():
    df = pd.read_excel(EXCEL_FILE)
    df = df.drop_duplicates(subset=['question'])

    # Load already tested questions
    if os.path.exists(OUTPUT_CSV):
        tested_df = pd.read_csv(OUTPUT_CSV, encoding='utf-8-sig')
        tested_questions = set(tested_df['question'].tolist())
    else:
        tested_questions = set()

    # Filter out already tested questions
    df_to_test = df[~df['question'].isin(tested_questions)]

    sample_size = min(TEST_AMOUNT, len(df_to_test))
    random_df = df_to_test.sample(n=sample_size, random_state=42)

    # If file does not exist, write header first
    write_header = not os.path.exists(OUTPUT_CSV)

    for idx, row in tqdm(random_df.iterrows(), total=len(random_df), desc="Processing Questions"):
        question = row['question']
        extractive_answer = row['extractive answer']
        abstractive_answer = row['abstractive answer']

        predicted_answer, question_type = call_chatbot_api(question)

        # Parse keywords
        if pd.notna(extractive_answer):
            keywords = [kw.strip() for kw in str(extractive_answer).split(',') if kw.strip()]
        else:
            keywords = []

        # Calculate metrics
        exact_match = keyword_exact_match(predicted_answer, keywords) if keywords else 0
        f1 = keyword_f1(predicted_answer, keywords) if keywords else 0.0
        bert = bert_score(predicted_answer, str(abstractive_answer)) if pd.notna(abstractive_answer) else 0.0

        # Store single result
        result = {
            'question': question,
            'extractive_answer': extractive_answer,
            'abstractive_answer': abstractive_answer,
            'predicted_answer': predicted_answer,
            'question_type': question_type,
            'keyword_exact_match': exact_match,
            'keyword_f1': f1,
            'bert_score': bert
        }

        # Append row immediately to CSV
        pd.DataFrame([result]).to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False, encoding='utf-8-sig')
        write_header = False  # Only write header once
        time.sleep(0.5)  # avoid overwhelming API

    print(f"\nProcessed {len(random_df)} questions. Results appended to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()