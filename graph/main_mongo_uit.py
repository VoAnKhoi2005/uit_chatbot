import phonlp
from tqdm import tqdm

from graph.src.db import *
from graph.src.triplet_extraction import *


def main():
    # Initialize SQLite connections
    process_conn, process_cursor = init_sqlite(r"E:\Github\uit_chatbot\graph\jupyter\uit_law.db")
    # law_conn, law_cursor = init_sqlite(r"E:\Github\uit_chatbot\graph\GTVT_law.db")

    # Initialize MongoDB
    mongo_client = init_mongo()
    if not mongo_client:
        print("Failed to connect to MongoDB. Exiting.")
        return
    db = mongo_client["KB_UIT"]

    # Initialize NLP models
    vncorenlp_client = init_vncorenlp(r"E:\Github\LawAssistant\triplet_extraction\VnCoreNLP-1.2")
    phoNLP_model = phonlp.load(save_dir=r"E:\Github\uit_chatbot\graph\phonlp")

    synonym_dict = load_synonym_dict(r"E:\Github\uit_chatbot\graph\listSameKey.txt")
    stopwords = load_stopwords(r"E:\Github\uit_chatbot\graph\stopwords.csv")

    no_triplet_csv_path = r"E:\Github\uit_chatbot\graph\no_triplets_uit_log_2.csv"
    no_triplet_file = open(no_triplet_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(no_triplet_file)
    csv_writer.writerow(["document_id", "document_number", "sentence"])

    logger, console_handler, file_handler = setup_logger(
        name="triplet_extraction",
        level=logging.DEBUG,
        log_to_file=True,
        file_path="logs/triplet_extraction.txt"
    )

    logger.removeHandler(console_handler)

    logger.info("Starting triplet extraction...")
    logger.debug("Debug mode enabled")

    try:
        reprocess_no_triplet = False
        if not reprocess_no_triplet:
            print("Đang xóa cơ sở dữ liệu cũ...")
            delete_all_mongo(db)
            print("Đang tạo indexes...")
            create_indexes(db)
            print("Bắt đầu trích xuất từ SQLite...")
            rows = extract_all_from_sqlite(process_cursor, "laws_process")
        else:
            # Nếu xử lý lại các câu chưa có triplet, đọc từ CSV log
            print("Đang đọc các câu chưa có triplet từ CSV...")
            rows = []
            with open(r"E:\Github\uit_chatbot\graph\no_triplets_uit_log_1.csv", "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({
                        "id": row["document_id"],
                        "so_hieu": row["document_number"],
                        "content": row["sentence"]
                    })

        print(f"Tìm thấy {len(rows)} hàng để xử lý.\n")

        for i, row in enumerate(tqdm(rows, desc="Đang xử lý văn bản", unit="văn bản")):
            sentence = row['content']
            if not sentence or not sentence.strip():
                continue

            triplets = triplet_extraction(
                text=sentence,
                vncorenlp_client=vncorenlp_client,
                phoNLP_model=phoNLP_model,
                stopwords=set(),
                logger=logger,
            )

            triplets_list = [
                {
                    "c1": c1,
                    "r": r,
                    "c2": c2
                }
                for (c1, r, c2) in triplets
                if c1 and r and c2
            ]

            doc_metadata = {
                'document_number': row.get('so_hieu', 'UNKNOWN'),
                'document_id': str(row.get('id', 'UNKNOWN'))
            }
            if triplets_list:
                try:
                    count = insert_triplet_batch_mongo(
                        db,
                        triplets_list=triplets_list,
                        metadata=doc_metadata,
                        synonym_dict=synonym_dict,
                    )
                    tqdm.write(f"[{i + 1}/{len(rows)}] Đã chèn {count} triplets cho doc_id {doc_metadata['document_id']}")
                except Exception as e:
                    tqdm.write(f"Lỗi khi chèn batch cho doc_id {doc_metadata['document_id']}: {e}")
            else:
                # Ghi lại các câu không trích xuất được vào log
                csv_writer.writerow([doc_metadata['document_id'], doc_metadata['document_number'], sentence])

    except Exception as e:
        print(f"Lỗi nghiêm trọng trong quá trình main: {e}")

    finally:
        if 'process_conn' in locals():
            process_conn.close()
        if 'mongo_client' in locals():
            mongo_client.close()
        if 'no_triplet_file' in locals():
            no_triplet_file.close()
        print("\nĐã đóng tất cả kết nối. Hoàn thành.")


if __name__ == "__main__":
    main()
