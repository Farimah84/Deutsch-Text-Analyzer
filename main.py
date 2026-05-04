import pandas as pd
import gradio as gr
import re
from collections import Counter

# ================================================================
# A Simple Dictionary (für Deutsch Wörter, Verben, Grammatik usw.)
# ================================================================

# Most common B1 Verbs
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

# Separable verb prefixes
separable_prefixes = ["auf", "an", "aus", "bei", "ein", "fest", "her", "hin", "los", "mit", "nach", "vor", "weg", "zu", "zurück", "zusammen"]

# Common prepositions
common_prepositions = [
    "in", "auf", "an", "unter", "über", "neben", "zwischen", "vor", "hinter",
    "aus", "bei", "mit", "nach", "von", "zu", "seit", "ohne", "durch", "für", "gegen"
]

# Articles (all lowercase for comparison)
articles = ["der", "die", "das", "den", "dem", "des"]

# Possessive pronouns (all lowercase)
possessive_pronouns = ["mein", "meine", "dein", "deine", "sein", "seine", "ihr", "ihre", "unser", "unsere", "euer", "eure"]

# Noun suffixes
noun_suffixes = ["heit", "keit", "ung", "schaft", "tion"]

def tokenize_simple(text):
    """Convert text to word list and remove simple punctuation"""
    text = re.sub(r'[.,!?;:()""\'\']', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.strip().split()
    return words

def detect_word_type(word, prev_word=None):
    """
    Identify word type without Spacy
    B1 rules and grammatical
    """
    word_lower = word.lower()
    
    # 1. Check for articles
    if word_lower in articles:
        return "ARTICLE"
    
    # 2. Check for possessive pronouns
    if word_lower in possessive_pronouns:
        return "POSS_PRONOUN"
    
    # 3. Check for definite noun suffixes
    if any(word_lower.endswith(suffix) for suffix in noun_suffixes):
        return "NOUN"
    
    # 4. Noun: Starts with capital letter (and not the beginning of sentence with verb)
    if word[0].isupper() and len(word) > 1:
        # Exception: Beginning of sentence and it's a verb
        if prev_word is None and word_lower in common_verbs:
            return "VERB"
        return "NOUN"
    
    # 5. Check for verbs
    if word_lower in common_verbs:
        return "VERB"
    
    # 6. Check for prepositions
    if word_lower in common_prepositions:
        return "ADP"
    
    # 7. Check for separable prefixes
    if word_lower in separable_prefixes:
        return "SEP_PREFIX"
    
    # 8. Check for adjectives (by suffix)
    if any(word_lower.endswith(suffix) for suffix in ["ig", "lich", "isch", "bar", "sam", "haft", "los"]):
        return "ADJ"
    
    return "OTHER"

def analyze_text_simple(text):
    """Simple version without Spacy"""
    sentences = text.split('.')
    words = tokenize_simple(text)
    
    # Analyze each word
    word_types = []
    prev_word = None
    for i, w in enumerate(words):
        wtype = detect_word_type(w, prev_word)
        word_types.append((w, wtype))
        prev_word = w
    
    # Statistics
    nouns = [w for w, t in word_types if t == "NOUN"]
    verbs = [w for w, t in word_types if t == "VERB"]
    prepositions = [w for w, t in word_types if t == "ADP"]
    prefixes = [w for w, t in word_types if t == "SEP_PREFIX"]
    
    stats = {
        "Total words": len(words),
        "Nouns (with capital letter)": len(nouns),
        "Verbs": len(verbs),
        "Prepositions": len(prepositions),
        "Separable prefixes": ", ".join(set(prefixes)) if prefixes else "None",
        "Most common noun": Counter(nouns).most_common(1)[0][0] if nouns else "---",
        "Most common verb": Counter(verbs).most_common(1)[0][0] if verbs else "---",
    }
    
    # Create dataframe
    df_stats = pd.DataFrame(list(stats.items()), columns=["Feature", "Value"])
    
    # Build highlighted HTML text
    html_parts = []
    for word, wtype in word_types:
        if wtype == "NOUN":
            color = "#4CAF50"  # green
            title = "Noun"
        elif wtype == "VERB":
            color = "#F44336"  # red
            title = "Verb"
        elif wtype == "ADP":
            color = "#FF9800"  # orange
            title = "Preposition"
        elif wtype == "SEP_PREFIX":
            color = "#9C27B0"  # purple
            title = "Separable prefix"
        elif wtype == "ARTICLE":
            color = "#FFC107"  # amber
            title = "Article"
        elif wtype == "POSS_PRONOUN":
            color = "#FF9800"  # orange
            title = "Possessive pronoun"
        elif wtype == "ADJ":
            color = "#2196F3"  # blue
            title = "Adjective"
        else:
            color = "#9E9E9E"  # gray
            title = "Other"
        html_parts.append(f'<span style="background-color:{color}; color:white; padding:2px 5px; border-radius:4px; margin:2px; display:inline-block;" title="{title}">{word}</span>')
    
    highlighted_text = " ".join(html_parts)
    
    # Cloze exercise
    cloze_exercises = []
    for sent in sentences[:3]:
        sent = sent.strip()
        if len(sent.split()) < 4:
            continue
        
        sent_words = sent.split()
        if len(sent_words) > 2:
            idx = min(2, len(sent_words)-1)
            removed = sent_words[idx]
            sent_words[idx] = "______"
            cloze_sent = " ".join(sent_words)
            cloze_exercises.append(f"• {cloze_sent}   [Answer: {removed}]")
    
    cloze_text = "\n\n".join(cloze_exercises) if cloze_exercises else "Not enough sentences for exercise."
    
    return highlighted_text, df_stats, cloze_text

# Gradio interface
def gradio_interface(text):
    highlight, table, cloze = analyze_text_simple(text)
    return highlight, table, cloze

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=10, placeholder="Enter German lyrics here...\nExample:\nDie Sonne scheint. Meine Freunde kommen. Das Glück ist groß."),
    outputs=[
        gr.HTML(label="Highlighted Text (Color Coded)"),
        gr.Dataframe(label="Statistics"),
        gr.Textbox(label="Fill-in-the-blank Exercise", lines=5)
    ],
    title="📝 German Text Analyzer • Level B1",
    description="🔹 Green: Nouns • Red: Verbs • Orange: Prepositions • Purple: Separable prefixes • Amber: Articles • Blue: Adjective"
)

if __name__ == "__main__":
    iface.launch()
