from langchain_core.documents import Document
import requests
from typing import List, Tuple, Optional
from bs4 import BeautifulSoup
from src.util.logging_util import get_logger
from src.util.common_util import ensure_nltk_resources
from nltk.tokenize import sent_tokenize

ensure_nltk_resources()


class WikiPageScrapper:
    """
    A processor for extracting, parsing, and chunking Wikipedia page content
    into structured Document objects for RAG systems.
    Attributes:
        max_tokens: Maximum number of tokens per chunk
        overlap_sents: Number of sentences to overlap between chunks
    """

    # Wikipedia heading hierarchy configuration
    HEADING_LEVELS = {
        "mw-heading2": {"level": 2, "parent": None},
        "mw-heading3": {"level": 3, "parent": 2},
        "mw-heading4": {"level": 4, "parent": 3},
        "mw-heading5": {"level": 5, "parent": 4},
        "mw-heading6": {"level": 6, "parent": 5}
    }

    TAG_CLASS_MAP = {
        "h2": "mw-heading2",
        "h3": "mw-heading3",
        "h4": "mw-heading4",
        "h5": "mw-heading5",
        "h6": "mw-heading6",
    }

    def __init__(self, excluded_section_headers: List[str], unwanted_tags_selectors: List[str], wiki_user_agent: str = "HybridRAGBot/1.0", max_tokens: int = 400, overlap_sents: int = 2):
        """
        Initialize the WikiPageProcessor.

        Args:
            max_tokens: Maximum number of tokens per chunk
            overlap_sents: Number of sentences to overlap between chunks
        """
        self.excluded_section_headers = excluded_section_headers
        self.unwanted_tags_selectors = unwanted_tags_selectors
        self.wiki_user_agent = wiki_user_agent
        self.max_tokens = max_tokens
        self.overlap_sents = overlap_sents
        self.logger = get_logger(__name__)


    def extract_content(self, url: str):
        """
        Extract and clean content from a Wikipedia page URL.

        Args:
            url: Wikipedia page URL

        Returns:
            BeautifulSoup content object or empty string if not found
        """
        header = {"User-Agent": self.wiki_user_agent}
        response = requests.get(url, headers=header)
        self.logger.info(f"Fetching content from URL: {url}")

        soup = BeautifulSoup(response.content, "lxml")

        # Remove unwanted tags
        for tag in soup.select(", ".join(self.unwanted_tags_selectors)):
            tag.decompose()

        contents = soup.select("div.mw-parser-output")
        self.logger.info(f"Extracted {len(contents)} content sections from the page.")

        # Identify main content by checking for the first <p> tag
        content = next((div for div in contents if div.find("p")), None)

        if content:
            self._remove_excluded_sections(content)

        return content if content else ""


    def _remove_excluded_sections(self, content):
        """
        Remove sections like References, External links, See also, etc.

        Args:
            content: BeautifulSoup content object (modified in-place)
        """
        for header_text in self.excluded_section_headers:
            header_tag = content.find(
                lambda tag: tag.name in list(self.TAG_CLASS_MAP.keys())
                and header_text in tag.get_text()
            )
            if header_tag:
                self.logger.debug(f"Removing section: {header_text} :: {header_tag.parent}")
                for sibling in header_tag.parent.find_next_siblings():
                    if self.TAG_CLASS_MAP.get(header_tag.name) in sibling.get("class", []):
                        self.logger.debug(f"Stopping removal at sibling: {sibling}")
                        break
                    sibling.decompose()
                header_tag.parent.decompose()


    def parse_sections(self, content) -> Tuple[Optional[str], List[dict]]:
        """
        Parse Wikipedia content into a hierarchical section structure (h2-h6).

        Args:
            content: BeautifulSoup content object

        Returns:
            Tuple of (short_description, sections_list)
        """
        sections = []
        abstract_texts = []
        short_desc = None

        # Track current headings at each level
        current_headings = {}

        sections.append({"abstract": abstract_texts})

        for child in content.children:
            if not child.get_text().strip():
                continue

            child_classes = child.attrs.get("class", [])

            # Check for Wiki hidden short description
            if "shortdescription" in child_classes:
                short_desc = child.get_text().strip()
                continue

            # Check for heading levels
            heading_found = False
            for class_name, config in self.HEADING_LEVELS.items():
                if class_name in child_classes:
                    level = config["level"]
                    parent_level = config["parent"]

                    # Create new heading
                    has_subsections = level < 6
                    new_heading = {
                        "title": child.get_text().strip(),
                        "texts": []
                    }
                    if has_subsections:
                        new_heading["subsections"] = []

                    # Attach to parent or sections
                    if parent_level and parent_level in current_headings:
                        current_headings[parent_level]["subsections"].append(new_heading)
                    else:
                        sections.append(new_heading)

                    # Update current heading at this level
                    current_headings[level] = new_heading

                    # Clear deeper level headings
                    for deeper_level in range(level + 1, 7):
                        current_headings.pop(deeper_level, None)

                    heading_found = True
                    break

            # If not a heading, it's text content
            if not heading_found:
                text = child.get_text().strip()
                # Assign text to the deepest current heading
                assigned = False
                for level in sorted(current_headings.keys(), reverse=True):
                    current_headings[level]["texts"].append(text)
                    assigned = True
                    break

                # If no heading exists, add to abstract
                if not assigned:
                    abstract_texts.append(text)

        self.logger.info(f"Total sections extracted: {len(sections)}")
        return short_desc, sections


    def chunk_sentences(self, text_blocks: List[str]) -> List[str]:
        """
        Chunk text blocks into overlapping sentence-based chunks.

        Args:
            text_blocks: List of text strings to chunk

        Returns:
            List of text chunks
        """
        text = " ".join(text_blocks)
        sentences = sent_tokenize(text)

        chunks = []
        current = []
        token_count = 0

        for sent in sentences:
            current.append(sent)
            token_count += len(sent.split())

            if token_count >= (self.max_tokens - 60):  # Buffer of 60 tokens
                chunks.append(" ".join(current))
                # Keep overlap sentences
                current = current[-self.overlap_sents:]
                token_count = sum(len(x.split()) for x in current)

        if current:
            chunks.append(" ".join(current))

        self.logger.debug(f"Total sentence chunks created: {len(chunks)}")
        return chunks


    def _process_section_recursive(
        self,
        section: dict,
        title: str,
        url: str,
        short_desc: Optional[str],
        docs: List[Document],
        section_hierarchy: Optional[List[str]] = None
    ):
        """
        Recursively process a section and its subsections at any depth.

        Args:
            section: The section dictionary with 'title', 'texts', and optionally 'subsections'
            title: Article title
            short_desc: Article short description (optional)
            docs: List to append Document objects to
            section_hierarchy: List of section titles from root to current
        """
        if section_hierarchy is None:
            section_hierarchy = []

        # Process texts directly under this section
        text_blocks = section.get("texts", [])
        if text_blocks:
            chunks = self.chunk_sentences(text_blocks)
            for idx, chunk in enumerate(chunks):
                # Build page content with full hierarchy
                content_lines = [f"Article: {title}"]

                # Only add topic line if short_desc exists
                if short_desc:
                    content_lines.append(f"Topic: {short_desc}")

                # Add section hierarchy
                if len(section_hierarchy) == 1:
                    content_lines.append(f"Section: {section_hierarchy[0]}")
                elif len(section_hierarchy) > 1:
                    content_lines.append(f"Section: {section_hierarchy[0]}")
                    for i, subsec_title in enumerate(section_hierarchy[1:], 1):
                        indent = "Sub" * i + "section"
                        content_lines.append(f"{indent}: {subsec_title}")

                content_lines.append("")  # Empty line before content
                content_lines.append(chunk)
                page_content = "\n".join(content_lines)

                # Build metadata
                metadata = {
                    "source": f"wiki_{title.replace(' ', '_')}",
                    "article": title,
                    "url": url,
                    "section": section_hierarchy[0] if section_hierarchy else None,
                    "subsection": section_hierarchy[1] if len(section_hierarchy) > 1 else None,
                    "chunk_id": "_".join([s.replace(' ', '_') for s in section_hierarchy]) + f"_{idx}"
                }

                # Only add short_description if it exists
                if short_desc:
                    metadata["short_description"] = short_desc

                # Add full hierarchy to metadata for deeper nesting
                if len(section_hierarchy) > 2:
                    metadata["hierarchy"] = " > ".join([str(s) for s in section_hierarchy])

                docs.append(Document(page_content=page_content, metadata=metadata))

        # Recursively process subsections
        for subsection in section.get("subsections", []):
            self._process_section_recursive(
                subsection, title, url, short_desc, docs,
                section_hierarchy + [subsection["title"]]
            )


    def build_documents(
        self,
        title: str,
        url: str,
        short_desc: Optional[str],
        sections: List[dict]
    ) -> List[Document]:
        """
        Build Document objects from parsed sections.

        Args:
            title: Article title
            short_desc: Article short description (optional)
            sections: List of section dictionaries

        Returns:
            List of Document objects
        """
        docs = []

        # 1. Abstract section
        abstract_texts = sections[0].get("abstract", [])
        if abstract_texts:
            abstract_chunks = self.chunk_sentences(abstract_texts)
            for idx, chunk in enumerate(abstract_chunks):
                # Build page content
                content_lines = [f"Article: {title}"]
                if short_desc:
                    content_lines.append(f"Topic: {short_desc}")
                content_lines.extend(["Section: Abstract", "", chunk])

                # Build metadata
                metadata = {
                    "source": f"wiki_{title.replace(' ', '_')}",
                    "article": title,
                    "url": url,
                    "section": "Abstract",
                    "subsection": None,
                    "chunk_id": f"Abstract_{idx}"
                }
                if short_desc:
                    metadata["short_description"] = short_desc

                docs.append(
                    Document(
                        page_content="\n".join(content_lines),
                        metadata=metadata
                    )
                )

        # 2. Process all main sections (h2 and beyond) recursively
        for section in sections[1:]:
            # Skip abstract and short_description entries
            if "abstract" in section or "short_description" in section:
                continue

            if "title" in section:
                self._process_section_recursive(
                    section, title, url, short_desc, docs,
                    [section["title"]]
                )

        self.logger.info(f"Total documents created: {len(docs)}")
        return docs


    def process_page(self, url: str, title: str) -> Tuple[Optional[str], List[Document]]:
        """
        Complete pipeline: extract, parse, and build documents from a Wikipedia page.

        Args:
            url: Wikipedia page URL
            title: Article title

        Returns:
            Tuple of (short_description, list_of_documents)
        """
        content = self.extract_content(url)
        if not content:
            self.logger.warning(f"No content extracted from {url}")
            return None, []

        short_desc, sections = self.parse_sections(content)
        documents = self.build_documents(title, url, short_desc, sections)

        return documents