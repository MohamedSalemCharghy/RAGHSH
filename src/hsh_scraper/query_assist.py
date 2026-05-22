"""Lokale Hilfen fuer Query-Bewertung, Umformulierung und Retrieval-Planung."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


CODE_PATTERN = re.compile(r"\b[A-Z]{1,5}-?\d{1,4}[A-Z]?\b")
ABBREVIATION_PATTERN = re.compile(r"\b[A-Z]{2,8}\b")
FACULTY_PATTERN = re.compile(
    r"\b(?:fak(?:ultat|ultaet|ultät)?|faculty)\s*([1-5]|i{1,3}|iv|v)\b|\bf\s*([1-5])\b",
    re.IGNORECASE,
)
FACULTY_CODE_PATTERN = re.compile(
    r"^(?:f|fak|fakultaet|fakultät|faculty)\s*(?:[1-5]|i{1,3}|iv|v)$",
    re.IGNORECASE,
)
PERSON_TITLE_PATTERN = re.compile(
    r"^(?:herr|frau|prof(?:essor)?|prof\.|dr|dr\.|dozent(?:in)?)$",
    re.IGNORECASE,
)

DEFINITION_MARKERS = [
    "bedeutung",
    "bedeutet",
    "steht fuer",
    "steht für",
    "legende",
    "abkuerzung",
    "abkürzung",
    "pruefungsform",
    "prüfungsform",
    "pruefungsformen",
    "prüfungsformen",
]

WORKFLOW_MARKERS = [
    "bewerbungsportal",
    "campusmanagement",
    "unterlagen hochladen",
    "digital einreichen",
    "online-antraege",
    "online-anträge",
    "online-antrag",
    "online antrag",
    "portal",
    "pruefungsanmeldung",
    "prüfungsanmeldung",
    "antragsformular",
    "formular",
]

CONTACT_MARKERS = [
    "kontakt",
    "ansprechperson",
    "ansprechpersonen",
    "studierendenservice",
    "studienberatung",
    "service center",
    "servicecenter",
    "telefon",
    "e-mail",
    "email",
    "zustaendig",
    "zuständig",
]

KNOWN_ABBREVIATIONS = {
    "ATPO",
    "BPO",
    "ECTS",
    "MPO",
    "PO",
    "SWS",
}

DEFINITION_FILLER_WORDS = {
    "das",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "ist",
    "von",
    "was",
}
PERSON_QUERY_FILLER_WORDS = {
    "wer",
    "ist",
    "sind",
    "war",
    "bitte",
    "eigentlich",
}
STUDY_FACT_MARKERS = (
    "abschluss",
    "zulassungsfrei",
    "zulassungsbeschraenkt",
    "zulassungsbeschränkt",
    "vorpraktikum",
    "regelstudienzeit",
    "studienbeginn",
    "sommersemester",
    "wintersemester",
    "wann kann man beginnen",
    "wie lange dauert",
    "wie lange",
)
STUDY_PROGRAM_GENERIC_TERMS = {
    "abschluss",
    "bachelor",
    "bachelorstudiengang",
    "hochschule",
    "hannover",
    "master",
    "masterstudiengang",
    "regelstudienzeit",
    "semester",
    "sommersemester",
    "studienbeginn",
    "studiengang",
    "studium",
    "vorpraktikum",
    "wann",
    "welchen",
    "welche",
    "welcher",
    "welches",
    "wie",
    "wintersemester",
    "zulassungsbeschraenkt",
    "zulassungsbeschränkt",
    "zulassungsfrei",
}
STUDY_PROGRAM_CLEANUP_WORDS = STUDY_PROGRAM_GENERIC_TERMS | {
    "beginnen",
    "dauert",
    "der",
    "die",
    "das",
    "gibt",
    "gibtes",
    "hat",
    "ist",
    "kann",
    "lange",
    "man",
    "und",
}


@dataclass
class QueryAssessment:
    original_query: str
    reasons: list[str]
    suggestions: list[str]
    clarification_hint: str
    detected_codes: list[str]
    detected_abbreviations: list[str]
    detected_faculties: list[str]
    intents: list[str] = field(default_factory=list)
    specificity: str = "klar"
    clarification_prompt: str = ""
    clarification_options: list[str] = field(default_factory=list)
    clarification_needed: bool = False

    @property
    def needs_user_choice(self) -> bool:
        return self.clarification_needed

    @property
    def retrieval_terms(self) -> list[str]:
        terms = [*self.detected_codes, *self.detected_abbreviations]
        if "definition" in self.intents and not terms:
            cleaned = _extract_definition_target(self.original_query)
            if cleaned:
                terms.append(cleaned)
        return _dedupe_keep_order(terms)


def build_plain_query_assessment(query: str) -> QueryAssessment:
    """Neutraler QueryAssessment für Läufe ohne Query Assist."""
    return QueryAssessment(
        original_query=query,
        reasons=[],
        suggestions=[],
        clarification_hint="",
        detected_codes=[],
        detected_abbreviations=[],
        detected_faculties=[],
        intents=[],
        specificity="query_assist_deaktiviert",
        clarification_prompt="",
        clarification_options=[],
        clarification_needed=False,
    )


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        key = " ".join(item.split()).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(" ".join(item.split()).strip())
    return unique


def _contains_any(text: str, needles: tuple[str, ...] | list[str]) -> bool:
    return any(needle in text for needle in needles)


def _roman_to_faculty(token: str) -> str | None:
    normalized = token.strip().lower()
    mapping = {
        "1": "I",
        "2": "II",
        "3": "III",
        "4": "IV",
        "5": "V",
        "i": "I",
        "ii": "II",
        "iii": "III",
        "iv": "IV",
        "v": "V",
    }
    roman = mapping.get(normalized)
    if not roman:
        return None
    return f"Fakultät {roman}"


def _detect_abbreviations(query: str, detected_codes: list[str]) -> list[str]:
    abbreviations = []
    upper_query = query.upper()
    for match in ABBREVIATION_PATTERN.findall(upper_query):
        if match in detected_codes:
            continue
        if match in KNOWN_ABBREVIATIONS:
            abbreviations.append(match)
    return _dedupe_keep_order(abbreviations)


def _detect_faculties(query: str) -> list[str]:
    faculties: list[str] = []
    for match in FACULTY_PATTERN.finditer(query):
        faculty = _roman_to_faculty(match.group(1) or match.group(2) or "")
        if faculty:
            faculties.append(faculty)
    return _dedupe_keep_order(faculties)


def _is_faculty_code(token: str) -> bool:
    compact = re.sub(r"\s+", "", token or "")
    return bool(FACULTY_CODE_PATTERN.fullmatch(compact))


def _extract_detected_codes(query: str) -> list[str]:
    return [
        code
        for code in CODE_PATTERN.findall(query.upper())
        if not _is_faculty_code(code)
    ]


def _extract_definition_target(query: str) -> str:
    lower = query.lower()
    for marker in (
        "was bedeutet",
        "was ist",
        "wofuer steht",
        "wofür steht",
        "steht fuer",
        "steht für",
        "definition von",
        "definition",
    ):
        lower = lower.replace(marker, " ")

    cleaned = re.sub(r"[^\wäöüß-]", " ", lower)
    tokens = [
        token
        for token in cleaned.split()
        if token not in DEFINITION_FILLER_WORDS
    ]
    return " ".join(tokens)


def _extract_person_terms(query: str) -> list[str]:
    cleaned = re.sub(r"[^\wäöüÄÖÜß.-]", " ", query)
    tokens = [token for token in cleaned.split() if token]
    person_terms: list[str] = []

    for token in tokens:
        normalized = token.casefold()
        if normalized in PERSON_QUERY_FILLER_WORDS:
            continue
        if PERSON_TITLE_PATTERN.fullmatch(token):
            continue
        if len(token) <= 1:
            continue
        person_terms.append(token)

    return person_terms[:3]


def _looks_person_lookup(query: str, lower: str) -> bool:
    if lower.startswith(("wer ist ", "wer war ", "wer sind ")):
        return bool(_extract_person_terms(query))
    if PERSON_TITLE_PATTERN.search(query):
        return bool(_extract_person_terms(query))
    return False


def _looks_study_program_facts(lower: str) -> bool:
    return _contains_any(lower, STUDY_FACT_MARKERS)


def _clean_study_program_candidate(value: str) -> str:
    cleaned = re.sub(r"[^\wäöüÄÖÜß.-]", " ", value)
    tokens = []
    for token in cleaned.split():
        normalized = token.casefold()
        if normalized in STUDY_PROGRAM_CLEANUP_WORDS:
            continue
        tokens.append(token)
    return " ".join(tokens[:4]).strip()


def _extract_study_program_focus(query: str) -> str:
    patterns = (
        r"\b(?:bachelor(?:studiengang)?|master(?:studiengang)?|studiengang)\s+(.+?)(?:,|\?| und | oder |$)",
        r"\b(?:hat|ist|bietet)\s+(.+?)(?:,|\?| und | oder |$)",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = _clean_study_program_candidate(match.group(1))
        if candidate:
            return candidate

    tokens = re.findall(r"[A-Za-zÄÖÜäöüß][\wÄÖÜäöüß.-]*", query)
    candidates: list[str] = []
    for index, token in enumerate(tokens):
        if index == 0 and token[:1].isupper():
            continue
        if not token[:1].isupper():
            continue
        if token.casefold() in STUDY_PROGRAM_GENERIC_TERMS:
            continue
        candidates.append(token)

    return " ".join(candidates[:3]).strip()


def _build_study_fact_queries(focus: str, lower: str) -> list[str]:
    focus = " ".join(focus.split()).strip()
    if not focus:
        return []

    queries = [
        f"{focus} Hochschule Hannover",
        f"{focus} Studiengang Hochschule Hannover",
    ]
    detected_aspects = 0

    if "zulass" in lower:
        detected_aspects += 1
        queries.extend([
            f"{focus} zulassungsfrei Hochschule Hannover",
            f"{focus} Zulassung Hochschule Hannover",
        ])

    if "vorprakt" in lower or "praktikum" in lower:
        detected_aspects += 1
        queries.extend([
            f"{focus} Vorpraktikum Hochschule Hannover",
            f"{focus} fachbezogene Ausbildung Hochschule Hannover",
        ])

    if _contains_any(lower, ("abschluss", "b.eng", "b. eng", "bachelor of", "master of")):
        detected_aspects += 1
        queries.extend([
            f"{focus} Abschluss Hochschule Hannover",
            f"{focus} Bachelor of Engineering Hochschule Hannover",
        ])

    if _contains_any(lower, ("regelstudienzeit", "wie lange", "dauert", "semester")):
        detected_aspects += 1
        queries.extend([
            f"{focus} Regelstudienzeit Hochschule Hannover",
            f"{focus} 7 Semester Hochschule Hannover",
        ])

    if _contains_any(lower, ("studienbeginn", "beginnen", "sommersemester", "wintersemester")):
        detected_aspects += 1
        queries.extend([
            f"{focus} Studienbeginn Hochschule Hannover",
            f"{focus} Sommersemester Wintersemester Hochschule Hannover",
        ])

    if detected_aspects >= 2:
        queries.append(f"{focus} Studiengang Informationen Hochschule Hannover")

    return _dedupe_keep_order(queries)


def _looks_definition_like(lower: str, detected_codes: list[str], detected_abbreviations: list[str]) -> bool:
    return (
        detected_codes
        or detected_abbreviations
        or _contains_any(lower, ("was ist", "was bedeutet", "definition", "steht fuer", "steht für", "abkuerzung", "abkürzung"))
    )


def _looks_workflow_like(lower: str) -> bool:
    return _contains_any(
        lower,
        (
            "hochladen",
            "unterlagen",
            "online-antrag",
            "online antrag",
            "online-antraege",
            "online-anträge",
            "formular",
            "antrag",
            "bewerben",
            "bewerbung",
            "pruefungsanmeldung",
            "prüfungsanmeldung",
            "pruefung anmelden",
            "prüfung anmelden",
            "campusmanagement",
            "portal",
        ),
    )


def _looks_exam_registration_like(lower: str) -> bool:
    return _contains_any(
        lower,
        (
            "pruefungsanmeldung",
            "prüfungsanmeldung",
            "pruefung anmelden",
            "prüfung anmelden",
            "campusmanagement",
        ),
    )


def _looks_contact_like(lower: str) -> bool:
    return _contains_any(
        lower,
        (
            "wen kontaktiere",
            "wen frage",
            "kontakt",
            "ansprech",
            "zustaendig",
            "zuständig",
            "telefon",
            "email",
            "e-mail",
            "beratung",
        ),
    )


def _looks_broad_workflow_question(
    lower: str,
    *,
    workflow_intent: bool,
    contact_intent: bool,
    exam_registration_intent: bool,
    detected_codes: list[str],
    detected_abbreviations: list[str],
    detected_faculties: list[str],
) -> bool:
    if exam_registration_intent:
        return False
    if detected_codes or detected_abbreviations or detected_faculties:
        return False
    if not (workflow_intent or contact_intent):
        return False

    generic_markers = (
        "online-antrag",
        "online antrag",
        "online-anträge",
        "online-antraege",
        "antrag",
        "formular",
        "online",
        "portal",
        "hochladen",
        "unterlagen",
        "einreichen",
        "wen kontaktiere",
        "kontakt",
    )
    if not _contains_any(lower, generic_markers):
        return False

    specific_markers = (
        "bewerbung",
        "bewerbungsportal",
        "beurlaubung",
        "pruefungsanmeldung",
        "prüfungsanmeldung",
        "rueckmeldung",
        "rückmeldung",
        "semesterbeitrag",
        "namensaenderung",
        "namensänderung",
        "anschrift",
        "adresse",
        "unfall",
        "erstsemester",
        "icms",
        "campusmanagement",
        "langzeitstudien",
        "studienberatung",
        "pruefungsverwaltung",
        "prüfungsverwaltung",
    )
    return not _contains_any(lower, specific_markers)


def _build_clarification_options(lower: str) -> list[str]:
    options: list[str] = []

    if _contains_any(lower, ("online-antrag", "online antrag", "online-anträge", "online-antraege", "antrag", "formular", "online")):
        options.extend([
            "Wie stelle ich einen Online-Antrag für eine Bewerbung oder das Hochladen von Unterlagen?",
            "Wie beantrage ich online eine Beurlaubung an der Hochschule Hannover?",
            "Wie funktioniert die Prüfungsanmeldung online im iCMS?",
            "Wie ändere ich Name oder Anschrift online an der Hochschule Hannover?",
        ])

    if _contains_any(lower, ("hochladen", "unterlagen", "einreichen")):
        options.extend([
            "Wo lade ich Bewerbungsunterlagen im Bewerbungsportal hoch?",
            "Wie reiche ich Unterlagen für eine Beurlaubung oder einen Antrag online ein?",
        ])

    if _contains_any(lower, ("wen kontaktiere", "kontakt", "ansprech")):
        options.extend([
            "Wen kontaktiere ich zur Bewerbung an der Hochschule Hannover?",
            "Wen kontaktiere ich zur Prüfungsverwaltung oder Prüfungsanmeldung?",
            "Wen kontaktiere ich zur Studienberatung?",
            "Wen kontaktiere ich bei Fragen zu iCMS oder Campusmanagement?",
        ])

    return _dedupe_keep_order(options)


def _add_definition_suggestions(
    suggestions: list[str],
    terms: list[str],
) -> None:
    for term in terms:
        suggestions.extend([
            f"{term} Bedeutung Hochschule Hannover",
            f"{term} steht für Hochschule Hannover",
            f"{term} Legende Hochschule Hannover",
        ])

        if term == "PO":
            suggestions.extend([
                "PO Prüfungsordnung Hochschule Hannover",
                "Prüfungsordnung Abkürzung PO Hochschule Hannover",
            ])
        elif term == "BPO":
            suggestions.extend([
                "BPO Besondere Prüfungsordnung Hochschule Hannover",
                "Besondere Prüfungsordnung Abkürzung BPO Hochschule Hannover",
            ])
        elif term == "ECTS":
            suggestions.extend([
                "ECTS European Credit Transfer System Hochschule Hannover",
                "ECTS Leistungspunkte Bedeutung Hochschule Hannover",
            ])
        elif term == "SWS":
            suggestions.extend([
                "SWS Semesterwochenstunden Hochschule Hannover",
                "SWS Abkürzung Semesterwochenstunden Hochschule Hannover",
            ])

        klausur_match = re.fullmatch(r"K(\d{2,3})", term)
        if klausur_match:
            minutes = klausur_match.group(1)
            suggestions.extend([
                f"{term} Klausur {minutes} Minuten",
                f"{term} bedeutet Klausur {minutes} Minuten",
            ])


def assess_query(query: str) -> QueryAssessment:
    query = " ".join(query.split()).strip()
    lower = query.lower()
    reasons: list[str] = []
    suggestions: list[str] = []
    intents: list[str] = []
    clarification_hint = "Du kannst optional Studiengang, Fakultät, Dokumenttyp oder Frist ergänzen."
    clarification_prompt = ""
    clarification_options: list[str] = []
    specificity = "klar"

    detected_codes = _extract_detected_codes(query)
    detected_abbreviations = _detect_abbreviations(query, detected_codes)
    detected_faculties = _detect_faculties(query)
    person_lookup = _looks_person_lookup(query, lower)
    person_terms = _extract_person_terms(query) if person_lookup else []
    study_program_intent = _looks_study_program_facts(lower)
    study_program_focus = _extract_study_program_focus(query) if study_program_intent else ""

    definition_intent = _looks_definition_like(lower, detected_codes, detected_abbreviations)
    workflow_intent = _looks_workflow_like(lower)
    contact_intent = _looks_contact_like(lower)
    exam_registration_intent = _looks_exam_registration_like(lower)
    broad_workflow_question = _looks_broad_workflow_question(
        lower,
        workflow_intent=workflow_intent,
        contact_intent=contact_intent,
        exam_registration_intent=exam_registration_intent,
        detected_codes=detected_codes,
        detected_abbreviations=detected_abbreviations,
        detected_faculties=detected_faculties,
    )

    if definition_intent:
        intents.append("definition")
        clarification_hint = (
            "Wenn du den Studiengang, das Modul oder den Dokumenttyp kennst, ergänze ihn für präzisere Treffer."
        )
        if detected_codes:
            reasons.append("Kurze Code- oder Abkürzungsfrage erkannt.")
        _add_definition_suggestions(
            suggestions,
            [*detected_codes, *detected_abbreviations],
        )

    if workflow_intent:
        intents.append("workflow")
        if exam_registration_intent:
            intents.append("exam_registration")
            suggestions.extend([
                "Prüfungsanmeldung Campusmanagement Hochschule Hannover",
                "Campusmanagement-Portal Prüfungsanmeldung Hochschule Hannover",
                "Prüfungen und Studium Campusmanagement Hochschule Hannover",
            ])
            for faculty in detected_faculties:
                suggestions.extend([
                    f"{faculty} Prüfungsanmeldung iCMS Hochschule Hannover",
                    f"{faculty} Prüfungsanmeldung Campusmanagement Hochschule Hannover",
                    f"{faculty} Prüfungsplan Prüfungsanmeldung Hochschule Hannover",
                ])
        else:
            suggestions.extend([
                "Online-Anträge und -Bewerbungen Hochschule Hannover",
                "Bewerbungsportal Hochschule Hannover",
            ])

    if contact_intent:
        intents.append("contact")
        suggestions.extend([
            "Kontakt Studieninteressierte Hochschule Hannover",
            "Kontakt Bewerbung Hochschule Hannover",
            "Studierendenservice Bewerbung Hochschule Hannover",
            "Studienberatung Bewerbung Hochschule Hannover",
            "Service Center Bewerbung Hochschule Hannover",
            "Ansprechperson Bewerbung Hochschule Hannover",
        ])

    if person_lookup:
        intents.append("person_lookup")
        reasons.append("Personenanfrage erkannt; Namen werden in den Suchpfaden beibehalten.")
        full_name = " ".join(person_terms)
        if full_name:
            suggestions.extend([
                f"{full_name} Hochschule Hannover",
                f"Personenfinder {full_name} Hochschule Hannover",
                f"{full_name} Kontakt Hochschule Hannover",
                f"Dozent {full_name} Hochschule Hannover",
            ])

    if study_program_intent:
        intents.append("study_program_facts")
        reasons.append("Studiengangsfrage mit konkreten Eckdaten erkannt.")
        suggestions.extend(
            _build_study_fact_queries(study_program_focus or query, lower)
        )

    if detected_faculties:
        reasons.append("Fakultätsangabe erkannt und für die Suche normalisiert.")
        for faculty in detected_faculties:
            suggestions.extend([
                f"{query} {faculty}",
                f"{faculty} Hochschule Hannover {query}",
            ])
            if any(term in lower for term in ("studiengang", "studiengänge", "studiengaenge")):
                suggestions.extend([
                    f"{faculty} Studiengänge Hochschule Hannover",
                    f"Studiengänge {faculty} Hochschule Hannover",
                ])

    if _contains_any(lower, ("hochladen", "unterlagen")):
        suggestions.extend([
            "Bewerbungsportal Unterlagen hochladen Hochschule Hannover",
            "Unterlagen digital im Bewerbungsportal einreichen Hochschule Hannover",
        ])

    if _contains_any(lower, ("online-antrag", "online antrag", "online-antraege", "online-anträge")):
        suggestions.extend([
            "Online-Anträge und -Bewerbungen Hochschule Hannover",
            "Online-Antragsformular Hochschule Hannover",
        ])

    if any(term in lower for term in ("unterbrechen", "pausieren", "pause")):
        intents.append("workflow")
        reasons.append("Alltagssprache erkannt, offizielle HsH-Begriffe könnten abweichen.")
        suggestions.extend([
            "Wie beantrage ich eine Beurlaubung an der Hochschule Hannover?",
            "Beurlaubung Studium Hochschule Hannover Fristen und Voraussetzungen",
        ])
        clarification_hint = "Falls bekannt, ergänze bitte deinen Studiengang oder deine Fakultät."

    if any(term in lower for term in ("rückmelde", "rueckmelde", "semesterbeitrag")):
        suggestions.extend([
            "Rückmeldung Fristen Hochschule Hannover",
            "Semesterbeitrag Rückmeldung Frist Hochschule Hannover",
        ])

    if "höhere fachsemester" in lower or "hoehere fachsemester" in lower:
        suggestions.extend([
            "Bewerbungsportal höhere Fachsemester Hochschule Hannover",
            "Bewerbung höhere Fachsemester und Quereinsteiger Hochschule Hannover",
        ])

    if "unfall" in lower:
        suggestions.extend([
            "Unfallversicherung Unfallmeldung Hochschule Hannover",
            "Wie melde ich einen Unfall an der Hochschule Hannover?",
        ])

    if any(
        term in lower
        for term in (
            "anschrift",
            "adresse",
            "namensänderung",
            "namensaenderung",
            "name ändern",
            "name aendern",
        )
    ):
        suggestions.extend([
            "Namensänderung und Änderung der Anschrift Hochschule Hannover",
            "Vordruck für Namensänderung und Änderung der Anschrift Hochschule Hannover",
        ])

    if any(
        term in lower
        for term in (
            "pruefungsform",
            "prüfungsform",
            "pruefungsformen",
            "prüfungsformen",
        )
    ):
        intents.append("definition")
        suggestions.extend([
            "Prüfungsformen Legende Hochschule Hannover",
            "Prüfungsform Bedeutung Hochschule Hannover",
            "Legende der Prüfungsformen Hochschule Hannover",
        ])

    if len(query) < 18 or len(query.split()) < 4:
        if definition_intent or not (workflow_intent or contact_intent):
            reasons.append("Die Frage ist sehr kurz und kann für die Suche mehrdeutig sein.")

    clarification_needed = False
    if broad_workflow_question:
        specificity = "allgemein"
        clarification_needed = True
        reasons.append(
            "Die Frage ist noch zu allgemein; an der HsH kann sie mehrere unterschiedliche Prozesse meinen."
        )
        clarification_prompt = (
            "Bitte wähle die genauere Richtung aus. Im Hintergrund werden trotzdem zusätzliche Suchvarianten vorbereitet."
        )
        clarification_hint = (
            "Du kannst zusätzlich Fakultät, Studiengang oder den genauen Antrag ergänzen."
        )
        clarification_options = _build_clarification_options(lower)
    elif definition_intent and len(query.split()) <= 2 and not (detected_codes or detected_abbreviations):
        specificity = "kurz"
        clarification_needed = True
    elif len(query.split()) < 2 and not (workflow_intent or contact_intent):
        specificity = "kurz"
        clarification_needed = True

    suggestions = _dedupe_keep_order([s for s in suggestions if s.lower() != query.lower()])
    reasons = _dedupe_keep_order(reasons)
    intents = _dedupe_keep_order(intents)
    clarification_options = _dedupe_keep_order(clarification_options)

    return QueryAssessment(
        original_query=query,
        reasons=reasons,
        suggestions=suggestions[:4],
        clarification_hint=clarification_hint,
        detected_codes=detected_codes,
        detected_abbreviations=detected_abbreviations,
        detected_faculties=detected_faculties,
        intents=intents,
        specificity=specificity,
        clarification_prompt=clarification_prompt,
        clarification_options=clarification_options[:4],
        clarification_needed=clarification_needed,
    )


def build_retrieval_queries(
    assessment: QueryAssessment,
    *,
    selected_query: str | None = None,
    clarification: str = "",
) -> list[str]:
    selected = (selected_query or assessment.original_query).strip()
    clarification = clarification.strip()
    primary = f"{selected} {clarification}".strip()
    lower_primary = primary.casefold()

    queries: list[str] = [primary]

    for suggestion in assessment.suggestions:
        if suggestion.lower() == selected.lower():
            continue
        queries.append(f"{suggestion} {clarification}".strip())
        if len(queries) >= 10:
            return _dedupe_keep_order(queries)[:10]

    if "definition" in assessment.intents:
        for term in assessment.retrieval_terms:
            queries.extend([
                f"{term} Bedeutung",
                f"{term} steht für",
                f"{term} Legende Hochschule Hannover",
                f"{term} Abkürzung Hochschule Hannover",
            ])

    if "workflow" in assessment.intents:
        if "exam_registration" in assessment.intents:
            queries.extend([
                f"{selected} Campusmanagement Hochschule Hannover",
                f"{selected} Prüfungen und Studium Hochschule Hannover",
            ])
            for faculty in assessment.detected_faculties:
                queries.extend([
                    f"{faculty} Prüfungsanmeldung iCMS Hochschule Hannover",
                    f"{faculty} Prüfungsanmeldung Campusmanagement Hochschule Hannover",
                    f"{faculty} Prüfungsplan Prüfungsanmeldung Hochschule Hannover",
                ])
        else:
            queries.extend([
                f"{selected} Portal Hochschule Hannover",
                f"{selected} Formular Hochschule Hannover",
            ])

    if "contact" in assessment.intents:
        queries.extend([
            f"{selected} Kontakt Hochschule Hannover",
            f"{selected} Ansprechperson Hochschule Hannover",
            f"{selected} Studienberatung Hochschule Hannover",
        ])

    if "person_lookup" in assessment.intents:
        person_terms = _extract_person_terms(selected)
        full_name = " ".join(person_terms)
        if full_name:
            queries.extend([
                f"{full_name} Hochschule Hannover",
                f"Personenfinder {full_name} Hochschule Hannover",
                f"{full_name} Kontakt Hochschule Hannover",
                f"Dozent {full_name} Hochschule Hannover",
            ])

    if "study_program_facts" in assessment.intents:
        focus = _extract_study_program_focus(selected) or _extract_study_program_focus(assessment.original_query)
        queries.extend(_build_study_fact_queries(focus or selected, lower_primary))

    for faculty in assessment.detected_faculties:
        queries.extend([
            f"{selected} {faculty}",
            f"{faculty} Hochschule Hannover {selected}",
        ])

    return _dedupe_keep_order(queries)[:10]


def apply_planner_guidance(
    assessment: QueryAssessment,
    planner_plan: dict[str, Any] | None,
) -> QueryAssessment:
    """Führt lokale Analyse und LLM-Planung zusammen."""
    if not planner_plan:
        return assessment

    reasons = list(assessment.reasons)
    suggestions = list(assessment.suggestions)
    clarification_options = list(assessment.clarification_options)
    clarification_hint = assessment.clarification_hint
    clarification_prompt = assessment.clarification_prompt
    clarification_needed = assessment.clarification_needed
    specificity = assessment.specificity

    reason = " ".join(str(planner_plan.get("reason") or "").split()).strip()
    if reason:
        reasons.append(f"LLM-Planer: {reason}")

    canonical_terms = [
        term
        for term in planner_plan.get("canonical_hsh_terms", [])
        if isinstance(term, str)
    ]
    source_type_hints = [
        hint
        for hint in planner_plan.get("source_type_hints", [])
        if isinstance(hint, str)
    ]
    for term in canonical_terms:
        suggestions.append(f"{term} Hochschule Hannover")
    for hint in source_type_hints:
        if hint == "contact":
            suggestions.append(f"{assessment.original_query} Kontakt Hochschule Hannover")
        elif hint == "portal":
            suggestions.append(f"{assessment.original_query} Portal Hochschule Hannover")
        elif hint == "form":
            suggestions.append(f"{assessment.original_query} Formular Hochschule Hannover")
        elif hint == "guide":
            suggestions.append(f"{assessment.original_query} Anleitung Hochschule Hannover")
        elif hint == "faq":
            suggestions.append(f"{assessment.original_query} FAQ Hochschule Hannover")
        elif hint == "regulation":
            suggestions.append(f"{assessment.original_query} Ordnung Hochschule Hannover")

    suggestions.extend(
        variant
        for variant in planner_plan.get("query_variants", [])
        if isinstance(variant, str)
    )

    if planner_plan.get("needs_clarification"):
        clarification_needed = True
        specificity = "allgemein"
        if planner_plan.get("clarification_prompt"):
            clarification_prompt = planner_plan["clarification_prompt"]
        clarification_options = [
            option
            for option in planner_plan.get("clarification_options", [])
            if isinstance(option, str)
        ] or clarification_options
        clarification_hint = (
            "Du kannst zusätzlich Fakultät, Studiengang, Portal oder den genauen Antrag ergänzen."
        )

    return QueryAssessment(
        original_query=assessment.original_query,
        reasons=_dedupe_keep_order(reasons),
        suggestions=_dedupe_keep_order(suggestions)[:6],
        clarification_hint=clarification_hint,
        detected_codes=assessment.detected_codes,
        detected_abbreviations=assessment.detected_abbreviations,
        detected_faculties=assessment.detected_faculties,
        intents=assessment.intents,
        specificity=specificity,
        clarification_prompt=clarification_prompt,
        clarification_options=_dedupe_keep_order(clarification_options)[:4],
        clarification_needed=clarification_needed,
    )
