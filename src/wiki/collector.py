import json
import random
import time
import requests
from urllib.parse import quote
from typing import Any, Literal
from src.util.logging_util import get_logger

class DataCollector:
    """
    A class for collecting Wikipedia pages data from various categories and random sources.

    Attributes:
        fixed_sample_size: Number of fixed pages to sample from categories
        fixed_max_pages: Maximum pages to fetch from all categories combined
        fixed_max_cat_pages: Maximum pages to fetch per category
        fixed_min_word_count: Minimum word count for page extracts
        random_sample_size: Number of random pages to collect
        random_max_pages: Maximum random pages to fetch
        random_min_page_size: Minimum page size in bytes for random pages
        random_seed: Seed for reproducible random selection
    """
    WIKI_BASE_URL = "https://en.wikipedia.org"
    WIKI_API_ENDPOINT = f"{WIKI_BASE_URL}/w/api.php"
    WIKI_PAGE_URL_TEMPLATE = f"{WIKI_BASE_URL}/wiki/{{page_title}}"

    def __init__(
        self,
        categories: list[str],
        fixed_sample_file: str,
        random_sample_file: str,
        wiki_user_agent: str = "HybridRAGBot/1.0",
        fixed_sample_size: int = 200,
        fixed_max_pages: int = 220,
        fixed_max_cat_pages: int = 20,
        fixed_min_word_count: int = 200,
        random_sample_size: int = 300,
        random_max_pages: int = 400,
        random_min_page_size: int = 8000,
        random_seed: int = 42,
    ):
        self.categories = categories
        self.fixed_sample_file = fixed_sample_file
        self.random_sample_file = random_sample_file
        self.wiki_user_agent = wiki_user_agent
        self.fixed_sample_size = fixed_sample_size
        self.fixed_max_pages = fixed_max_pages
        self.fixed_max_cat_pages = fixed_max_cat_pages
        self.fixed_min_word_count = fixed_min_word_count
        self.random_sample_size = random_sample_size
        self.random_max_pages = random_max_pages
        self.random_min_page_size = random_min_page_size
        self.random_seed = random_seed
        self.logger = get_logger(__name__)


    def collect_data(self, refresh_fixed_wiki_pages=False, refresh_random_wiki_pages=True):
        self.logger.info("Starting data collection process...")
        try:
            fixed_wiki_pages = None
            if refresh_fixed_wiki_pages:
                self.logger.info("Refreshing fixed wiki pages...")
                fixed_wiki_pages = self.generate_fixed_wiki_pages(
                    categories=self.categories,
                    sample_size=self.fixed_sample_size,
                    max_pages=self.fixed_max_pages,
                    max_cat_pages=self.fixed_max_cat_pages,
                    min_word_count=self.fixed_min_word_count,
                    random_seed=self.random_seed,
                )
                # Write to a JSON file
                self.logger.info(f"Writing fixed wiki pages to {self.fixed_sample_file}")
                with open(self.fixed_sample_file, "w") as f:
                    json.dump(fixed_wiki_pages, f, indent=4)

            if refresh_random_wiki_pages:
                self.logger.info("Refreshing random wiki pages...")
                if fixed_wiki_pages is None:
                    self.logger.info(
                        f"Loading fixed wiki pages from {self.fixed_sample_file} to fetch page ids for duplicate avoidance."
                    )
                    # Load JSON file to get the list of page ids
                    with open(self.fixed_sample_file, "r") as f:
                        fixed_wiki_pages = json.load(f)

                fixed_page_ids = [page["page_id"] for page in fixed_wiki_pages["pages"]]
                self.logger.info(f"Total fixed wiki pages loaded: {len(fixed_page_ids)}")

                random_wiki_pages = self.fetch_random_wiki_pages(
                    fixed_page_ids,
                    sample_size=self.random_sample_size,
                    max_pages=self.random_max_pages,
                    min_page_size=self.random_min_page_size,
                    random_seed=self.random_seed,
                )

                # Write to a JSON file
                self.logger.info(f"Writing random wiki pages to {self.random_sample_file}")
                with open(self.random_sample_file, "w") as f:
                    json.dump(random_wiki_pages, f, indent=4)

            if not refresh_fixed_wiki_pages and not refresh_random_wiki_pages:
                self.logger.info("No refresh tasks were specified.")
            else:
                self.logger.info("Data collection process completed successfully.")

        except Exception as e:
            self.logger.error(f"Error during data collection: {e}")
            raise e


    def generate_fixed_wiki_pages(
        self,
        categories: list[str],
        sample_size=200,
        max_pages=220,
        max_cat_pages=20,
        min_word_count=200,
        random_seed=42,
    ) -> dict[str, Any]:
        diverse_pages = []
        for category in categories:
            fetched_pages = self.fetch_wiki_pages_for_category(
                category,
                max_pages=max_cat_pages,
                min_extract_length=min_word_count,
                random_seed=random_seed,
            )
            print(f"Category: {category} | Pages fetched: {len(fetched_pages)}")
            diverse_pages.extend(fetched_pages)
            if len(diverse_pages) >= max_pages:
                break
        print(f"Total Pages fetched: {len(diverse_pages)}")
        fixed_wiki_pages = {
            "created_at": round(time.time() * 1000),
            "pages": random.sample(diverse_pages, sample_size),
        }
        return fixed_wiki_pages


    def fetch_wiki_pages_for_category(
        self,
        category: str,
        page_limit=100,
        max_pages=40,
        min_extract_length=200,
        random_seed=42,
    ) -> list[dict[str, Any]]:
        headers = {"User-Agent": self.wiki_user_agent}
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": page_limit,  # Maximum number of pages to fetch
            "cmnamespace": 0,  # Only fetch pages from the main/article namespace
            "format": "json",
        }
        self.logger.info(f"Fetching URLs for category: {category}")
        response = requests.get(self.WIKI_API_ENDPOINT, params=params, headers=headers)
        self.logger.debug(f"Request URL: {response.url}")
        self.logger.debug(
            f"Fetched pages with category:{category} |  Response Status Code: {response.status_code}"
        )
        data = response.json()
        # logger.debug(f"Response Data: {data}")

        pages = []
        category_members = data.get("query", {}).get("categorymembers", [])
        random.seed(random_seed)
        random.shuffle(category_members)  # Shuffle to get random pages
        for member in category_members:
            page_id = member["pageid"]
            page_title = member["title"]
            page_ref = page_title.replace(" ", "_")
            extract = self.fetch_extract_for_page(page_ref)
            self.logger.debug(
                f"Processing page title: {page_title} | Extract length: {len(extract)}"
            )
            if len(extract) >= min_extract_length:
                pages.append(
                    {
                        "page_id": page_id,
                        "page_title": page_title,
                        "extract_length": len(extract),
                        "url": self.WIKI_PAGE_URL_TEMPLATE.format(page_title=quote(page_ref)),
                        "category": category,
                    }
                )
            else:
                self.logger.info(
                    f"Skipping page: {page_title} due to insufficient extract length"
                )
            if len(pages) >= max_pages:
                break
            time.sleep(0.15)  # To respect API rate limits
        return pages


    def fetch_extract_for_page(self, page_title) -> Any | Literal['']:
        """
        Retrieve introductory text extract for a Wikipedia page.

        Args:
            page_title: Page title (can include underscores)

        Returns:
            Plain text extract or empty string if not found
        """
        headers = {"User-Agent": self.wiki_user_agent}
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": page_title,
            "format": "json",
        }
        self.logger.debug(f"Fetching extract for page: {page_title}")
        response = requests.get(self.WIKI_API_ENDPOINT, params=params, headers=headers)
        self.logger.debug(f"Request URL: {response.url}")
        self.logger.info(
            f"Fetched extract for page:{page_title} |  Response Status Code: {response.status_code}"
        )
        data = response.json()
        # logger.debug(f"Response Data: {data}")

        pages = data.get("query", {}).get("pages", {})
        for _, page_data in pages.items():
            extract = page_data.get("extract", "")
            return extract
        return ""

    def fetch_random_wiki_pages(
        self,
        fixed_page_ids: list[int],
        sample_size=300,
        max_pages=400,
        min_page_size=8000,
        random_seed=42,
    ) -> dict[str, Any]:
        """
        Fetch random Wikipedia pages excluding specified page IDs.

        Args:
            fixed_page_ids: Page IDs to exclude (from fixed collection)
            sample_size: Target number of unique pages (default: 300)
            max_pages: Max pages to request from API (default: 400)
            min_page_size: Min page size in bytes (default: 8000)
            random_seed: Seed for reproducibility (default: 42)

        Returns:
            Dict with created_at timestamp and list of random pages
        """
        headers = {"User-Agent": self.wiki_user_agent}
        params = {
            "action": "query",
            "list": "random",
            "rnnamespace": 0,  # Only fetch pages from the main/article namespace
            "rnlimit": max_pages,
            "rnminsize": min_page_size,
            "format": "json",
        }
        self.logger.info(f"Fetching {max_pages} random Wikipedia pages")
        response = requests.get(self.WIKI_API_ENDPOINT, params=params, headers=headers)
        self.logger.debug(f"Request URL: {response.url}")
        self.logger.debug(
            f"Fetched random pages | Response Status Code: {response.status_code}"
        )
        data = response.json()
        # logger.debug(f"Response Data: {data}")
        random_wiki_pages = {
            "created_at": round(time.time() * 1000),
        }
        pages = []
        random_page_ids = set(fixed_page_ids)
        random_pages = data.get("query", {}).get("random", [])
        random.seed(random_seed)
        random.shuffle(random_pages)  # Shuffle to randomize list
        duplicate_tracker = 0
        for member in random_pages:
            page_id = member["id"]
            page_title = member["title"]
            random_page_ids.add(page_id)
            if len(random_page_ids) - len(fixed_page_ids) - len(pages) == 0:
                duplicate_tracker += 1
                self.logger.info(
                    f"Skipping duplicate page: {page_title} (ID: {page_id})"
                )
                continue
            else:
                pages.append(
                    {
                        "page_id": page_id,
                        "page_title": page_title,
                        "url": self.WIKI_PAGE_URL_TEMPLATE.format(
                            page_title=quote(page_title.replace(" ", "_"))
                        ),
                    }
                )

            if len(random_page_ids) == sample_size + len(fixed_page_ids):
                self.logger.info("Reached desired sample size of random pages.")
                break

        random_wiki_pages["pages"] = pages
        self.logger.info(
            f"Total random pages fetched: {len(pages)} with {duplicate_tracker} duplicates skipped."
        )
        return random_wiki_pages