from graph.src.db import *
from graph.src.triplet_extraction import *


def main():
    conn, cursor = init_sqlite(r"./law.db")
    gpt_client = init_gpt()
    vncorenlp_client = init_vncorenlp(r"E:\Github\LawAssistant\triplet_extraction\VnCoreNLP-1.2")

    # rows = extract_random_from_sqlite(cursor, True)
    rows = extract_from_sqlite(cursor, "b7fee66c5d809a5fd7023b6810338639e4b19563187a93e21e17d4e21e92da77", True)
    law = ""
    title = ""
    for r in rows:
        title += r['title'] + " "
        if not (r['title'].strip().startswith("Chương") or r['title'].strip().startswith("Điều")):
            law += r['content'] + "\n"

    so_hieu = r['so_hieu']
    id = r['id']
    print(id)
    print(so_hieu + " " + title)
    print(clean_text(law))

    # system_prompt_rewrite, user_prompt_rewrite = law_sentence_completion_prompt(law, so_hieu)
    # rewrite_sentence = generate_response_gpt_4_1_mini(system_prompt_rewrite, user_prompt_rewrite, gpt_client)
    # print("Rewrite sentence: ")
    # print(rewrite_sentence)
    # print()

    rewrite_sentence = """
    Giấy phép lái xe bao gồm các hạng sau đây.  
    Hạng DE cấp cho người lái các loại xe ô tô quy định cho giấy phép lái xe hạng D.  
    Hạng DE cấp cho người lái các loại xe ô tô kéo rơ moóc có khối lượng toàn bộ theo thiết kế trên 750 kg.  
    Hạng DE cấp cho người lái xe ô tô chở khách nối toa.
    """

    print("Processing sentence...")
    result = extract_triplet(rewrite_sentence, vncorenlp_client, True)
    for r in result:
        print(r)
    pass

if __name__ == "__main__":
    main()