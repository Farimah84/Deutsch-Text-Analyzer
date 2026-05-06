import pandas as pd
import gradio as gr
import re
from collections import Counter

# ================================================================
# GERMAN GRAMMAR LISTS (B1 Level)
# ================================================================

common_verbs = [
    "bin", "bist", "ist", "sind", "seid", "war", "waren",
    "habe", "hast", "hat", "haben", "hattet", "hatte",
    "werde", "wirst", "wird", "werden", "wurde", "wurden",
    "gehe", "gehst", "geht", "gehen", "ging", "gingen", "gegangen",
    "komme", "kommst", "kommt", "kommen", "kam", "kamen", "gekommen",
    "sehe", "siehst", "sieht", "sehen", "sah", "sahen", "gesehen",
    "esse", "isst", "essen", "aß", "aßen", "gegessen",
    "trinke", "trinkst", "trinkt", "trinken", "trank", "getrunken",
    "schreibe", "schreibst", "schreibt", "schreiben", "schrieb", "geschrieben",
    "lese", "liest", "lesen", "las", "lasen", "gelesen"
]

separable_prefixes = ["auf", "an", "aus", "bei", "ein", "fest", "her", "hin", "los", "mit", "nach", "vor", "weg", "zu", "zurück", "zusammen"]

common_prepositions = [
    "in", "auf", "an", "unter", "über", "neben", "zwischen", "vor", "hinter",
    "aus", "bei", "mit", "nach", "von", "zu", "seit", "ohne", "durch", "für", "gegen",
    "wegen", "trotz", "innerhalb", "außerhalb"
]

# New: Genitive prepositions
genitive_prepositions = ["wegen", "trotz", "innerhalb", "außerhalb"]

# New: Dative prepositions (zu, von, bei)
dative_prepositions = ["zu", "von", "bei", "mit", "nach", "aus"]

# New: Accusative prepositions (für, um, durch, gegen, ohne)
accusative_prepositions = ["für", "um", "durch", "gegen", "ohne"]

articles = ["der", "die", "das", "den", "dem", "des"]
possessive_pronouns = ["mein", "meine", "dein", "deine", "sein", "seine", "ihr", "ihre", "unser", "unsere", "euer", "eure"]

subj_conjunctions = ["weil", "dass", "wenn", "während", "obwohl", "da", "bevor", "nachdem", "ob", "damit"]
coord_conjunctions = ["und", "oder", "aber", "denn", "sondern", "doch", "trotzdem", "deshalb", "darum", "deswegen", "daher", "nämlich", "außerdem"]
question_words = ["wer", "was", "wie", "wo", "woher", "wohin", "wann", "warum", "wieso", "welch"]

noun_suffixes = ["heit", "keit", "ung", "schaft", "tion", "ismus", "ität", "ik", "chen", "lein"]  # +chen for diminutive

# New: Superlative endings
superlative_endings = ["ste", "sten", "stes", "ste"]


def tokenize_simple(text):
    text = re.sub(r'[.,!?;:()""\'\']', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().split()


def detect_tense(verb, word, sentence_context=""):
    word_lower = word.lower()
    
    # Plusquamperfekt: "hatte/war + Partizip II"
    if "hatte" in sentence_context or "war" in sentence_context:
        if word_lower.startswith("ge") or word_lower.endswith("t"):
            return "Plusquamperfekt"
    
    # Perfekt
    if word_lower.startswith("ge") and len(word_lower) > 3:
        return "Perfekt"
    
    # Präteritum Passiv: "wurde + Partizip II"
    if "wurde" in sentence_context and (word_lower.startswith("ge") or word_lower.endswith("t")):
        return "Präteritum Passiv"
    
    # Präteritum
    praeteritum_forms = ["war", "waren", "hatte", "hatten", "ging", "gingen", "kam", "kamen", "sah", "sahen", "aß", "aßen", "trank", "tranken", "schrieb", "schrieben", "las", "lasen", "wurde", "wurden"]
    if word_lower in praeteritum_forms:
        return "Präteritum"
    
    # Präsens
    if word_lower in common_verbs:
        return "Präsens"
    
    return "Unknown"


def detect_grammar_patterns(words, idx, word, prev_word, next_word):
    """Detect special grammar patterns"""
    patterns = []
    word_lower = word.lower()
    
    # 1. zu + Infinitiv (e.g., "zu lernen")
    if word_lower == "zu" and next_word and next_word.lower() in common_verbs:
        patterns.append("zu + Infinitiv")
    
    # 2. Werden + Infinitiv (Future I: "werde lernen")
    if word_lower in ["werde", "wirst", "wird", "werden"] and next_word and next_word.lower() in common_verbs:
        patterns.append("Futur I (werden + Infinitiv)")
    
    # 3. Werden + Partizip II (Passiv Präsens)
    if word_lower in ["wird", "werden", "werde", "wirst"] and next_word and (next_word.lower().startswith("ge") or next_word.lower().endswith("t")):
        patterns.append("Passiv Präsens (werden + Partizip II)")
    
    # 4. Wurde + Partizip II (Passiv Präteritum)
    if word_lower == "wurde" and next_word and (next_word.lower().startswith("ge") or next_word.lower().endswith("t")):
        patterns.append("Passiv Präteritum (wurde + Partizip II)")
    
    # 5. Je...desto pattern
    if word_lower == "je" and any(w.lower() == "desto" for w in words[idx:idx+5]):
        patterns.append("je...desto (Komparativsatz)")
    
    # 6. Um...zu pattern
    if word_lower == "um" and next_word and next_word.lower() == "zu":
        patterns.append("um...zu (Finaler Infinitivsatz)")
    
    # 7. Nicht nur...sondern auch
    if word_lower == "nicht" and next_word and next_word.lower() == "nur":
        patterns.append("nicht nur...sondern auch")
    
    # 8. Sowohl...als auch
    if word_lower == "sowohl" and any(w.lower() == "auch" for w in words[idx:idx+5]):
        patterns.append("sowohl...als auch")
    
    # 9. Adjektiv mit Superlativ (e.g., "schönste", "größten")
    if any(word_lower.endswith(ending) for ending in superlative_endings):
        patterns.append("Superlativ (Adjektivendung)")
    
    # 10. Diminutive "-chen" (e.g., "Häuschen")
    if word_lower.endswith("chen") and len(word_lower) > 4:
        patterns.append("Diminutiv (-chen)")
    
    # 11. Genitive after preposition (wegen + Genitiv)
    if prev_word and prev_word.lower() in genitive_prepositions:
        if word[0].isupper() or word_lower.endswith("s") or word_lower.endswith("es"):
            patterns.append(f"Genitiv (nach {prev_word})")
    
    # 12. Dative after preposition (zu, von, bei + Dativ)
    if prev_word and prev_word.lower() in dative_prepositions:
        patterns.append(f"Dativ (nach {prev_word})")
    
    # 13. Accusative after preposition (für, um, durch + Akkusativ)
    if prev_word and prev_word.lower() in accusative_prepositions:
        patterns.append(f"Akkusativ (nach {prev_word})")
    
    # 14. Ohne Artikel (null article)
    if prev_word is None or prev_word.lower() not in articles + possessive_pronouns:
        if word[0].isupper() and word not in ["Ich", "Du", "Er", "Sie", "Es", "Wir", "Ihr"]:
            patterns.append("Ohne Artikel (Nullartikel)")
    
    return patterns


def detect_word_type(word, prev_word=None, next_word=None, sentence_context=""):
    word_lower = word.lower()
    
    # Special conjunctions and connectors
    if word_lower in ["trotzdem", "deshalb", "darum", "deswegen", "daher", "nämlich", "außerdem"]:
        return {"type": "KONNEKTOR", "info": f"Konjunktionaladverb ({word})", "tense": None}
    
    if word_lower in ["während"]:
        if next_word and next_word[0].isupper():
            return {"type": "PREPOSITION", "info": "Präposition (während + Genitiv)", "tense": None}
        else:
            return {"type": "SUB_CONJ", "info": "Nebensatzkonjunktion (während)", "tense": None}
    
    if word_lower in subj_conjunctions:
        return {"type": "SUB_CONJ", "info": "Nebensatz einleitend", "tense": None}
    
    if word_lower in coord_conjunctions:
        return {"type": "CONJ", "info": "Hauptsatz verbindend", "tense": None}
    
    if word_lower in question_words:
        return {"type": "QUESTION", "info": "Fragewort", "tense": None}
    
    if word_lower in articles:
        return {"type": "ARTICLE", "info": "Begleiter", "tense": None}
    
    if word_lower in possessive_pronouns:
        return {"type": "POSSESSIVE", "info": "Possessivpronomen", "tense": None}
    
    if word_lower in common_prepositions:
        case_info = ""
        if word_lower in genitive_prepositions:
            case_info = " (Genitiv)"
        elif word_lower in dative_prepositions:
            case_info = " (Dativ)"
        elif word_lower in accusative_prepositions:
            case_info = " (Akkusativ)"
        return {"type": "PREPOSITION", "info": f"Präposition{case_info}", "tense": None}
    
    if word[0].isupper() and len(word) > 1:
        if prev_word is None and word_lower in common_verbs:
            return {"type": "VERB", "info": "Verb (am Satzanfang)", "tense": detect_tense(word, word, sentence_context)}
        return {"type": "NOUN", "info": "Substantiv", "tense": None}
    
    if any(word_lower.endswith(suffix) for suffix in noun_suffixes):
        return {"type": "NOUN", "info": "Substantiv (Suffix)", "tense": None}
    
    if word_lower in common_verbs:
        tense = detect_tense(word, word, sentence_context)
        return {"type": "VERB", "info": f"Verb ({tense})", "tense": tense}
    
    if word_lower in separable_prefixes:
        return {"type": "SEP_PREFIX", "info": "Trennbarer Präfix", "tense": None}
    
    if any(word_lower.endswith(suffix) for suffix in ["ig", "lich", "isch", "bar", "sam", "haft", "los"]):
        return {"type": "ADJECTIVE", "info": "Adjektiv", "tense": None}
    
    if any(word_lower.endswith(ending) for ending in superlative_endings):
        return {"type": "ADJECTIVE", "info": "Adjektiv (Superlativ)", "tense": None}
    
    if word_lower.isdigit():
        return {"type": "NUMBER", "info": "Zahl", "tense": None}
    
    return {"type": "OTHER", "info": "Anderes", "tense": None}


def analyze_lyrics(text):
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    words = tokenize_simple(text)
    
    word_analysis = []
    for idx, w in enumerate(words):
        prev_word = words[idx-1] if idx > 0 else None
        next_word = words[idx+1] if idx + 1 < len(words) else None
        sentence_context = " ".join(words[max(0, idx-5):min(len(words), idx+5)])
        
        analysis = detect_word_type(w, prev_word, next_word, sentence_context)
        grammar_patterns = detect_grammar_patterns(words, idx, w, prev_word, next_word)
        
        info_text = analysis["info"]
        if grammar_patterns:
            info_text += f" | {', '.join(grammar_patterns)}"
        
        word_analysis.append({
            "word": w,
            "type": analysis["type"],
            "info": info_text,
            "tense": analysis["tense"],
            "patterns": grammar_patterns
        })
    
    type_counts = Counter([wa["type"] for wa in word_analysis])
    tense_counts = Counter([wa["tense"] for wa in word_analysis if wa["tense"]])
    
    examples = {}
    for wa in word_analysis:
        if wa["type"] not in examples:
            examples[wa["type"]] = []
        if len(examples[wa["type"]]) < 3:
            examples[wa["type"]].append(wa["word"])
    
    stats_data = []
    for type_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        stats_data.append({
            "Category (German)": get_german_name(type_name),
            "Category (English)": type_name,
            "Count": count,
            "Examples": ", ".join(examples.get(type_name, [])[:3])
        })
    
    if tense_counts:
        stats_data.append({"Category (German)": "--- Zeiten ---", "Category (English)": "--- Tenses ---", "Count": "", "Examples": ""})
        for tense, count in tense_counts.items():
            stats_data.append({
                "Category (German)": f"  {tense}",
                "Category (English)": tense,
                "Count": count,
                "Examples": ""
            })
    
    # New: Special pattern statistics
    all_patterns = []
    for wa in word_analysis:
        all_patterns.extend(wa["patterns"])
    pattern_counts = Counter(all_patterns)
    if pattern_counts:
        stats_data.append({"Category (German)": "--- Besondere Strukturen ---", "Category (English)": "--- Special Patterns ---", "Count": "", "Examples": ""})
        for pattern, count in pattern_counts.items():
            stats_data.append({
                "Category (German)": f"  {pattern}",
                "Category (English)": pattern,
                "Count": count,
                "Examples": ""
            })
    
    df_stats = pd.DataFrame(stats_data)
    
    html_parts = []
    for wa in word_analysis:
        tooltip = f"{wa['word']} → {wa['type']}: {wa['info']}"
        html_parts.append(f'<span title="{tooltip}" style="cursor: help; border-bottom: 1px dotted #999; margin: 0 2px;">{wa["word"]}</span>')
    
    highlighted_text = " ".join(html_parts)
    
    cloze_exercises = []
    for sent in sentences[:3]:
        sent_words = sent.split()
        if len(sent_words) < 4:
            continue
        if len(sent_words) > 2:
            idx = min(2, len(sent_words)-1)
            removed = sent_words[idx]
            sent_words[idx] = "______"
            cloze_sent = " ".join(sent_words)
            cloze_exercises.append(f"• {cloze_sent}\n  → (Answer: {removed})")
    
    cloze_text = "\n\n".join(cloze_exercises) if cloze_exercises else "Not enough sentences for exercise."
    
    return highlighted_text, df_stats, cloze_text


def get_german_name(type_name):
    names = {
        "NOUN": "Substantiv (Nomen)",
        "VERB": "Verb",
        "PREPOSITION": "Präposition",
        "ARTICLE": "Artikel",
        "POSSESSIVE": "Possessivpronomen",
        "ADJECTIVE": "Adjektiv",
        "SUB_CONJ": "Nebensatzkonjunktion",
        "CONJ": "Nebenordnende Konjunktion",
        "KONNEKTOR": "Konjunktionaladverb",
        "QUESTION": "Fragewort",
        "SEP_PREFIX": "Trennbarer Präfix",
        "NUMBER": "Zahl",
        "OTHER": "Andere"
    }
    return names.get(type_name, type_name)


def gradio_interface(text):
    highlight, table, cloze = analyze_lyrics(text)
    return highlight, table, cloze

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=12, placeholder="Enter German lyrics here...\n\nExample:\nJe mehr ich lerne, desto besser werde ich. Ich versuche, Deutsch zu lernen, weil es Spaß macht."),
    outputs=[
        gr.HTML(label="📖 Text with Tooltips"),
        gr.Dataframe(label="📊 Statistics"),
        gr.Textbox(label="✏️ Fill-in-the-blank Exercises", lines=8)
    ],
    title="📝 German Lyrics Analyzer • B1+ Level",
    description="Hover over any word to see detailed grammar information, including special patterns like 'zu + Infinitiv', 'Passiv', 'Superlativ', 'je...desto', and more."
)

if __name__ == "__main__":
    iface.launch()
