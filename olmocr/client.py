"""
Simple Python client for olmOCR that processes PDF bytes and returns text.
All processing happens in memory without saving to disk.
"""

import asyncio
import base64
import json

import fitz
import httpx

from olmocr.prompts import PageResponse, build_no_anchoring_v4_yaml_prompt
from olmocr.train.dataloader import FrontMatterParser


class OlmOCRClient:
    """
    Client for extracting text from PDFs using olmOCR.
    All processing happens in memory without saving to disk.

    Example:
        >>> client = OlmOCRClient(server_url="http://localhost:30024/v1")
        >>> with open("document.pdf", "rb") as f:
        ...     pdf_bytes = f.read()
        >>> text = client.extract_text(pdf_bytes)
        >>> print(text)
    """

    def __init__(
        self,
        server_url: str,
        api_key: str | None = None,
        target_longest_image_dim: int = 1288,
        model_name: str = "olmocr",
        max_retries: int = 8,
    ):
        """
        Initialize the olmOCR client.

        Args:
            server_url: URL of the olmOCR server (e.g., "http://localhost:30024/v1")
            api_key: Optional API key for authenticated servers
            target_longest_image_dim: Target dimension for the longest side of rendered pages
            model_name: Model name to use (default: "olmocr")
            max_retries: Maximum number of retries per page
        """
        self.server_url = server_url
        self.api_key = api_key
        self.target_longest_image_dim = target_longest_image_dim
        self.model_name = model_name
        self.max_retries = max_retries
        self._completion_url = f"{self.server_url.rstrip('/')}/chat/completions"
        self._client = httpx.AsyncClient()

        self._page_semaphore = asyncio.Semaphore(100)

    async def __process_page(self, page_num: int, fitz_doc: fitz.Document) -> PageResponse:
        """Process a single page and return its text."""
        MODEL_MAX_CONTEXT = 16384
        TEMPERATURE_BY_ATTEMPT = [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]
        exponential_backoffs = 0
        attempt = 0

        page = fitz_doc.load_page(page_num - 1)
        width, height = page.rect.width, page.rect.height
        longest_dim = max(width, height)
        dpi = int(self.target_longest_image_dim * 72 / longest_dim)
        pix = page.get_pixmap(dpi=dpi)
        image_base64 = base64.b64encode(pix.tobytes()).decode("utf-8")

        query = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_no_anchoring_v4_yaml_prompt()},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ],
            "max_tokens": 8000,
            "temperature": 0.0,
        }

        while attempt < self.max_retries:
            lookup_attempt = min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)

            query["temperature"] = TEMPERATURE_BY_ATTEMPT[lookup_attempt]

            try:
                response = await self._client.post(
                    self._completion_url,
                    json=query,
                )

                if response.status_code == 400:
                    raise ValueError(f"Got BadRequestError from server, skipping this response")
                elif response.status_code == 429:
                    raise ConnectionError("Too many requests, doing exponential backoff")
                elif response.status_code == 500:
                    raise ValueError(f"Got InternalServerError from server, skipping this response")
                elif response.status_code != 200:
                    raise ValueError(f"Error http status {response.status_code}")

                base_response_data = response.json()

                if base_response_data["usage"]["total_tokens"] > MODEL_MAX_CONTEXT:
                    raise ValueError(f"Response exceeded model_max_context of {MODEL_MAX_CONTEXT}, cannot use this response")

                if base_response_data["choices"][0]["finish_reason"] != "stop":
                    raise ValueError("Response did not finish with reason code 'stop', cannot use this response")

                model_response_markdown = base_response_data["choices"][0]["message"]["content"]

                parser = FrontMatterParser(front_matter_class=PageResponse)
                front_matter, text = parser._extract_front_matter_and_text(model_response_markdown)
                page_response = parser._parse_front_matter(front_matter, text)

                if not page_response.is_rotation_valid and attempt < self.max_retries - 1:
                    raise ValueError(f"invalid_page rotation, skipping this response")

                return page_response
            except (ConnectionError, OSError, asyncio.TimeoutError):
                sleep_delay = 10 * (2**exponential_backoffs)
                exponential_backoffs += 1
                await asyncio.sleep(sleep_delay)
            except asyncio.CancelledError:
                raise
            except json.JSONDecodeError:
                attempt += 1
            except ValueError:
                attempt += 1
            except Exception:
                attempt += 1

        raise ValueError(f"Failed to process page {page_num}")

    async def _process_page(self, page_num: int, fitz_doc: fitz.Document) -> PageResponse:
        async with self._page_semaphore:
            return await self.__process_page(page_num, fitz_doc)

    async def extract_text(self, pdf_bytes: bytes) -> list[PageResponse]:
        """
        Extract text from PDF bytes using olmOCR.
        All processing happens in memory without saving to disk.

        Args:
            pdf_bytes: The PDF file content as bytes

        Returns:
            List of PageResponse objects for all pages in the PDF

        Example:
            >>> client = OlmOCRClient("http://localhost:30024/v1")
            >>> pdf_bytes = open("document.pdf", "rb").read()
            >>> text = await client.extract_text(pdf_bytes)
            >>> print(text)
        """
        # Create PdfReader once to avoid recreating it for each page
        # pdf_reader = PdfReader(BytesIO(pdf_bytes))
        # num_pages = pdf_reader.get_num_pages()

        with fitz.open(stream=pdf_bytes, filetype="pdf") as fitz_doc:
            # Process all pages concurrently
            tasks = [self._process_page(page_num, fitz_doc) for page_num in range(1, fitz_doc.page_count + 1)]
            results = await asyncio.gather(*tasks)

        return results

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
