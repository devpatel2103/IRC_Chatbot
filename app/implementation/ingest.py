import re
import os
import json
import sys
import xml.etree.ElementTree as ET
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
IRC_PATH = "../Internal Revenue Code/IRC xml.xml"
VECTOR_DB_PATH = "../irc_xml_vectordb"

# ── Namespace ─────────────────────────────────────────────────────────────────
NS = "{http://xml.house.gov/schemas/uslm/1.0}"

# ── Tags to skip entirely (including all their children) ─────────────────────
SKIP_TAGS = {
    "notes", "note", "sourceCredit",
    "toc", "layout",
    "ref", "date", "meta",
}


def t(tag):
    """Strip namespace from a tag."""
    return tag.replace(NS, "")


def build_parent_map(root):
    """Build a child → parent mapping for the entire tree."""
    parent_map = {}
    for parent in root.iter():
        for child in parent:
            parent_map[child] = parent
    return parent_map


def get_ancestor_labels(section, parent_map):
    """
    Walk up the tree from a section element and collect
    subtitle, chapter, subchapter, part, subpart labels.
    """
    ancestors = {
        "subtitle":   "",
        "chapter":    "",
        "subchapter": "",
        "part":       "",
        "subpart":    "",
    }

    el = section
    while el is not None:
        el = parent_map.get(el)
        if el is None:
            break
        tag = t(el.tag)
        if tag in ancestors and not ancestors[tag]:
            # Get num + heading for this ancestor
            num_el     = el.find(f"{NS}num")
            heading_el = el.find(f"{NS}heading")
            num     = (num_el.text     or "").strip() if num_el     is not None else ""
            heading = (heading_el.text or "").strip() if heading_el is not None else ""
            ancestors[tag] = f"{num} {heading}".strip()

    return ancestors


def extract_legal_text(element):
    """
    Recursively extract clean legal text, skipping all noise tags.
    Preserves reading order including tail text.
    """
    tag = t(element.tag)

    if tag in SKIP_TAGS:
        return []

    texts = []

    if element.text and element.text.strip():
        texts.append(element.text.strip())

    for child in element:
        child_tag = t(child.tag)
        if child_tag not in SKIP_TAGS:
            texts.extend(extract_legal_text(child))
        if child.tail and child.tail.strip():
            if child_tag not in SKIP_TAGS:
                texts.append(child.tail.strip())

    return texts


def parse_and_clean(xml_path):
    print(f"Parsing {xml_path} ...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    main  = root.find(f"{NS}main")
    if main is None:
        print("ERROR: Could not find <main> element.")
        sys.exit(1)

    # Build parent map once — used for ancestor lookups
    print("  Building parent map ...")
    parent_map = build_parent_map(root)

    documents = []
    skipped   = 0

    # Iterate ALL section tags in document order
    all_sections = list(root.iter(f"{NS}section"))
    print(f"  Found {len(all_sections)} total <section> tags")

    for section in all_sections:
        identifier = section.get("identifier", "")

        # Only process real IRC sections (/us/usc/t26/sXXX)
        if "/s" not in identifier:
            skipped += 1
            continue

        section_num = identifier.split("/s")[-1]

        # ── Heading ───────────────────────────────────────────────────
        heading_el = section.find(f"{NS}heading")
        heading    = ""
        if heading_el is not None and heading_el.text:
            heading = heading_el.text.strip()

        # ── Clean legal text ──────────────────────────────────────────
        legal_parts = extract_legal_text(section)
        legal_text  = " ".join(legal_parts).strip()
        legal_text  = re.sub(r'\s+', ' ', legal_text)

        if len(legal_text) < 50:
            skipped += 1
            continue

        # ── Ancestor context ──────────────────────────────────────────
        ancestors = get_ancestor_labels(section, parent_map)

        # ── Bake section ID into text for retrieval ───────────────────

        documents.append(Document(
            page_content=legal_text,
            metadata={
                "section":    section_num,
                "heading":    heading,
                "identifier": identifier,
                "subtitle":   ancestors["subtitle"],
                "chapter":    ancestors["chapter"],
                "subchapter": ancestors["subchapter"],
                "part":       ancestors["part"],
                "subpart":    ancestors["subpart"],
                "source":     "IRC",
            }
        ))

    print(f"  Parsed  : {len(documents)} sections")
    print(f"  Skipped : {skipped} (no valid identifier or too short)")
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400,
    )
    chunks = splitter.split_documents(documents)
    
    # Prepend section identifier to EVERY chunk so BM25 can always find it
    for chunk in chunks:
        section = chunk.metadata.get("section", "")
        heading = chunk.metadata.get("heading", "")
        prefix  = f"§ {section} {heading}\n"
        
        # Only add prefix if it's not already there (first chunk already has it)
        if not chunk.page_content.startswith(f"§ {section}"):
            chunk.page_content = prefix + chunk.page_content
    
    print(f"  Chunked : {len(chunks)} chunks from {len(documents)} sections")
    return chunks


def save_chunks(chunks):
    output_path = "../Internal Revenue Code/irc_chunks.json"
    data = [
        {"text": c.page_content, "metadata": c.metadata}
        for c in chunks
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    print(f"Saved {len(data)} chunks to {output_path}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(VECTOR_DB_PATH):
        Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=VECTOR_DB_PATH)

    return vectorstore


def main():
    documents = parse_and_clean(IRC_PATH)

    if not documents:
        print("ERROR: No documents parsed.")
        sys.exit(1)

    chunks = chunk_documents(documents)
    save_chunks(chunks)

if __name__ == "__main__":
    main()
