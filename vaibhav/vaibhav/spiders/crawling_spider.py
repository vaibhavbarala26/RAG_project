import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from bs4 import BeautifulSoup
import re
import itertools
import json

# --- Configuration for different libraries ---
# This is the key to making the spider reusable.
# We define the unique settings for each documentation site here.
LIBRARY_CONFIGS = {
    'pandas': {
        'allowed_domains': ['pandas.pydata.org'],
        'start_urls': ['https://pandas.pydata.org/docs/user_guide/index.html'],
        'link_extractor_allow': r'/docs/user_guide/',
        'main_content_selector': ('article', {'class': 'bd-article'}),
    },
    'numpy': {
        'allowed_domains': ['numpy.org'],
        'start_urls': ['https://numpy.org/doc/stable/user/index.html'], # User guide has better narrative content for RAG
        'link_extractor_allow': r'/doc/stable/user/',
        'main_content_selector': ('article', {'class': 'bd-article'}),
    },
    'scikit-learn': {
        'allowed_domains': ['scikit-learn.org'],
        'start_urls': ['https://scikit-learn.org/stable/user_guide.html'],
        'link_extractor_allow': r'/stable/modules/',
        'main_content_selector': ('article', {'class': 'bd-article'}),
    },
    'tensorflow': {
        'allowed_domains': ['www.tensorflow.org'],
        'start_urls': ['https://www.tensorflow.org/api_docs/python/tf'],
        'link_extractor_allow': r'/api_docs/python/tf/',
        'main_content_selector': ('div', {'class': 'devsite-article-body'}),
    },
    'pytorch': {
        'allowed_domains': ['pytorch.org'],
        'start_urls': ['https://pytorch.org/docs/stable/index.html'],
        'link_extractor_allow': r'/docs/stable/',
        'main_content_selector': ('main', {'role': 'main'}),
    },
    'matplotlib':{
        'allowed_domains':['matplotlib.org'],
        'start_urls':['https://matplotlib.org/stable/users/index'],
        'link_extractor_allow':r'stable/users/',
        'main_content_selector':('article',{'class':'bd-article'})
    },
        'xgboost': {
        'allowed_domains': ['xgboost.readthedocs.io'],
        'start_urls': ['https://xgboost.readthedocs.io/en/stable/get_started.html'],
        'link_extractor_allow': r'/en/stable/',
        'main_content_selector': ('div', {'role': 'main'}),
    },
    'lightgbm': {
        'allowed_domains': ['lightgbm.readthedocs.io'],
        'start_urls': ['https://lightgbm.readthedocs.io/en/latest/index.html'],
        'link_extractor_allow': r'/en/latest/',
        'main_content_selector': ('div', {'role': 'main'}),
    },

    # --- Natural Language Processing ---
    'huggingface_transformers': {
        'allowed_domains': ['huggingface.co'],
        'start_urls': ['https://huggingface.co/docs/transformers/index'],
        'link_extractor_allow': r'/docs/transformers/',
        'main_content_selector': ('div', {'class': 'prose-doc'}),
    },
    'nltk': {
        'allowed_domains': ['www.nltk.org'],
        'start_urls': ['https://www.nltk.org/book/'],
        'link_extractor_allow': r'book/',
        'main_content_selector': ('body'),
    },
    'streamlit': {
        'allowed_domains': ['docs.streamlit.io'],
        'start_urls': ['https://docs.streamlit.io/'],
        'link_extractor_allow': r'/',
        'main_content_selector': ('main', {}),
    },
    'seaborn': {
        'allowed_domains': ['seaborn.pydata.org'],
        'start_urls': ['https://seaborn.pydata.org/tutorial.html'],
        'link_extractor_allow': r'/tutorial/',
        'main_content_selector': ('article', {'class': 'bd-article'}),
    },
}


class DocsSpider(CrawlSpider):
    name = 'docs_spider'

    # Class attributes for chunking
    global_chunk_counter = itertools.count()
    MAX_TOKENS = 300

    def __init__(self, library='pandas', *args, **kwargs):
        super(DocsSpider, self).__init__(*args, **kwargs)

        if library not in LIBRARY_CONFIGS:
            raise ValueError(f"Unknown library: {library}. Available options are: {list(LIBRARY_CONFIGS.keys())}")

        self.config = LIBRARY_CONFIGS[library]
        self.library_name = library
        
        # Apply the configuration from the selected library
        self.allowed_domains = self.config['allowed_domains']
        self.start_urls = self.config['start_urls']
        
        # Dynamically create the crawling rule based on the config
        DocsSpider.rules = (
            Rule(
                LinkExtractor(allow=self.config['link_extractor_allow']),
                callback='parse_item',
                follow=True
            ),
        )
        
        # Re-compile the rules with the new configuration
        super(DocsSpider, self)._compile_rules()

    def _clean_text(self, text: str) -> str:
        """Remove unwanted characters and normalize whitespace."""
        if not text:
            return ""
        text = text.replace("¶", "")
        text = re.sub(r'In \[\d+\]: ', '', text)
        text = re.sub(r'Out\[\d+\]: ', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _table_to_text(self, table_soup) -> str:
        """Convert an HTML table to a markdown-like text format."""
        rows = []
        for tr in table_soup.find_all('tr'):
            cells = [self._clean_text(td.get_text()) for td in tr.find_all(['td', 'th'])]
            rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _get_heading_level(self, tag_name):
        """Extract heading level (e.g., 1 for <h1>, 2 for <h2>) from a tag name."""
        if tag_name and tag_name.startswith('h') and len(tag_name) == 2:
            try:
                return int(tag_name[1])
            except ValueError:
                return None
        return None

    def _split_into_subchunks(self, text, max_tokens):
        """Split large text into smaller chunks based on word count."""
        words = text.split()
        sub_chunks = []
        current_chunk = []
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_tokens:
                sub_chunks.append(" ".join(current_chunk))
                current_chunk = []
        if current_chunk:
            sub_chunks.append(" ".join(current_chunk))
        return sub_chunks

    def parse_item(self, response):
        """
        Parses a documentation page, splitting it into semantic chunks based on headings.
        """
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Use the configured selector to find the main content area of the page
        tag, attrs = self.config['main_content_selector']
        main_content = soup.find(tag, attrs)

        if not main_content:
            return

        headings = main_content.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        
        prev_chunk_id = None
        for i, heading in enumerate(headings):
            current_content_list = [self._clean_text(heading.get_text())]
            current_heading_level = self._get_heading_level(heading.name)

            # Iterate through siblings of the heading to gather content
            for sibling in heading.find_next_siblings():
                # Stop if we hit the next heading of the same or higher level
                if i + 1 < len(headings) and sibling == headings[i + 1]:
                    break
                sibling_level = self._get_heading_level(sibling.name)
                if sibling_level is not None and sibling_level <= current_heading_level:
                    break
                
                # Extract text based on the tag type
                content_text = ""
                if sibling.name == "p":
                    content_text = sibling.get_text()
                elif sibling.name in ["div", "pre"] and "highlight" in ' '.join(sibling.get('class', [])):
                    pre_tag = sibling.find("pre")
                    if pre_tag:
                        content_text = pre_tag.get_text()
                elif sibling.name in ["ul", "ol"]:
                    items = [f"- {li.get_text(strip=True)}" for li in sibling.find_all("li", recursive=False)]
                    content_text = "\n".join(items)
                elif sibling.name == "table":
                    content_text = self._table_to_text(sibling)

                cleaned_text = self._clean_text(content_text)
                if cleaned_text:
                    current_content_list.append(cleaned_text)

            # Filter out chunks that only contain a heading and no other content
            if len(current_content_list) <= 1:
                continue

            final_chunk_content = "\n\n".join(current_content_list)
            section_title = self._clean_text(heading.get_text())

            if final_chunk_content:
                sub_chunks = self._split_into_subchunks(final_chunk_content, self.MAX_TOKENS)
                for idx, sub in enumerate(sub_chunks):
                    chunk_id = next(self.global_chunk_counter)
                    yield {
                        "chunk_id": f"chunk_{chunk_id}",
                        "url": response.url,
                        "title": section_title,
                        "page_title": response.css("title::text").get(),
                        "breadcrumbs": section_title,
                        "content": sub,
                        "prev_chunk_id": f"chunk_{prev_chunk_id}" if prev_chunk_id else None,
                        "next_chunk_id": None,
                        "type": "section"
                    }
                    prev_chunk_id = chunk_id
