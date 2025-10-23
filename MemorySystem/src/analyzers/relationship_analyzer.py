from typing import List, Tuple


def naive_relationships(sentences: List[str]) -> List[Tuple[str, str, str]]:
    """Очень простой анализатор связей: ищет повторяющиеся термины между предложениями
    и добавляет связи similar_to.
    Возвращает список (term, term, relation_type).
    """
    terms = [set(s.lower().split()) for s in sentences]
    rels = []
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            common = terms[i].intersection(terms[j])
            if len(common) >= 2:
                rels.append((f"sent{i}", f"sent{j}", "similar_to"))
    return rels


def advanced_relationships(text: str) -> List[Tuple[str, str, str, float]]:
    """Продвинутый эвристический анализатор отношений.
    Возвращает список кортежей (src_concept, dst_concept, relation_type, weight).
    Типы: causes, part_of, similar_to, related.
    Вес оценивается по наличию маркеров и длине совпадающих токенов.
    """
    import re
    normalized = text.lower()
    # Разбивка на простые "концепты" — сущ. слова (грубая эвристика)
    tokens = [t for t in re.findall(r"[a-zа-яё0-9_-]+", normalized) if len(t) > 2]
    unique = list(dict.fromkeys(tokens))[:20]

    rels: List[Tuple[str, str, str, float]] = []
    # Маркеры причинно-следственных отношений
    cause_markers = ["потому что", "ведет к", "приводит к", "из-за", "causes", "leads to", "due to"]
    part_markers = ["часть", "состоит из", "входит в", "part of", "includes"]
    similar_markers = ["похож", "аналог", "similar", "like"]

    def score_pair(a: str, b: str) -> float:
        # Чем длиннее общая подстрока, тем выше базовый вес (очень грубо)
        common = len(os_longest_common_substring(a, b))
        return min(1.0, 0.3 + 0.1 * common)

    window_text = normalized
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            a, b = unique[i], unique[j]
            base = score_pair(a, b)
            rtype = "related"
            weight = base
            for m in cause_markers:
                if m in window_text:
                    rtype, weight = "causes", max(weight, 0.8)
                    break
            if rtype == "related":
                for m in part_markers:
                    if m in window_text:
                        rtype, weight = "part_of", max(weight, 0.7)
                        break
            if rtype == "related":
                for m in similar_markers:
                    if m in window_text:
                        rtype, weight = "similar_to", max(weight, 0.6)
                        break
            rels.append((a, b, rtype, round(weight, 3)))
    return rels


def os_longest_common_substring(a: str, b: str) -> str:
    # Простая LCS по подстроке (O(n*m)) — достаточно для коротких токенов
    best = ""
    for i in range(len(a)):
        for j in range(len(b)):
            k = 0
            while i + k < len(a) and j + k < len(b) and a[i + k] == b[j + k]:
                k += 1
            if k > len(best):
                best = a[i:i + k]
    return best

