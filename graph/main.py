import phonlp
from tqdm import tqdm

from graph.src.db import *
from graph.src.triplet_extraction import *


def main():
    process_conn, process_cursor = init_sqlite(r"E:\Github\uit_chatbot\graph\process_law.db")
    law_conn, law_cursor = init_sqlite(r"E:\Github\uit_chatbot\graph\GTVT_law.db")
    neo4j_driver = init_neo4j(
        uri="neo4j://127.0.0.1:7687",
        username="neo4j",
        password="1234567890",
    )
    vncorenlp_client = init_vncorenlp(r"E:\Github\LawAssistant\triplet_extraction\VnCoreNLP-1.2")
    phoNLP_model = phonlp.load(save_dir=r"E:\Github\uit_chatbot\graph\phonlp")

    try:
        with neo4j_driver.session(database="ontology") as session:
            print("Đang xóa cơ sở dữ liệu cũ...")
            delete_all(session)

            print("Bắt đầu trích xuất từ SQLite...")
            rows = extract_all_from_sqlite(process_cursor, "laws")
            print(f"Tìm thấy {len(rows)} hàng để xử lý.\n")

            for i, row in enumerate(tqdm(rows, desc="Đang xử lý văn bản", unit="văn bản")):
                sentence = row['content']

                if not sentence or not sentence.strip():
                    continue

                segmented_text = vncorenlp_client.word_segment(sentence)
                annotation = phoNLP_model.annotate(text=segmented_text[0])
                df = parsing_result(annotation)
                result = process_sentence(df)

                if result:
                    doc_metadata = {
                        'document_number': row.get('so_hieu', 'UNKNOWN'),
                        'document_id': str(row.get('id', 'UNKNOWN'))
                    }

                    triplets_list = [
                        {"c1": c1, "r": r, "c2": c2}
                        for (c1, r, c2) in result
                        if c1 and r and c2
                    ]

                    if triplets_list:
                        try:
                            session.execute_write(
                                insert_triplet_batch,
                                triplets_list=triplets_list,
                                metadata=doc_metadata
                            )
                            tqdm.write(f"[{i + 1}/{len(rows)}] Đã chèn {len(triplets_list)} triplets cho doc_id {doc_metadata['document_id']}")
                        except Exception as e:
                            tqdm.write(f"Lỗi khi chèn batch cho doc_id {doc_metadata['document_id']}: {e}")
    except Exception as e:
        print(f"Lỗi nghiêm trọng trong quá trình main: {e}")
    finally:
        if 'process_conn' in locals():
            process_conn.close()
        if 'law_conn' in locals():
            law_conn.close()
        if 'neo4j_driver' in locals():
            neo4j_driver.close()
        print("\nĐã đóng tất cả kết nối. Hoàn thành.")


if __name__ == "__main__":
    main()
