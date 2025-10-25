def law_sentence_completion_prompt(sentence, so_hieu):
    with open(r"system_prompt.md", "r", encoding="utf-8") as f:
        content = f.read()
    system_prompt_rewrite = content

    user_prompt_rewrite = (
        f"Ngữ cảnh: Bộ luật số {so_hieu} trong luật Việt Nam.\n"
        f"Nhiệm vụ: Viết lại câu sau để hoàn chỉnh cấu trúc với đầy đủ chủ ngữ và vị ngữ, giữ nguyên ý nghĩa. "
        f"Mỗi câu xuất ra phải là một câu đơn đầy đủ (một dòng một câu nếu có nhiều câu).\n"
        f'Câu cần viết lại: "{sentence}"'
    )

    return system_prompt_rewrite, user_prompt_rewrite
