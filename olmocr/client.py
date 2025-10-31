"""
Simple Python client for olmOCR that processes PDF bytes and returns text.
All processing happens in memory without saving to disk.
"""

import asyncio
import base64
import json
import subprocess
from io import BytesIO

import httpx
from PIL import Image
from pypdf import PdfReader

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

    async def _apost(self, json_data: dict) -> tuple[int, bytes]:
        """Simple async HTTP POST implementation using httpx."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = await self._client.post(
            self._completion_url,
            json=json_data,
            headers=headers,
        )
        return response.status_code, response.content

    def _get_pdf_page_dimensions(self, pdf_reader: PdfReader, page_num: int) -> tuple[float, float]:
        """Get the dimensions of a PDF page in points."""
        page = pdf_reader.pages[page_num - 1]  # pages are 0-indexed
        mediabox = page.mediabox
        width = abs(mediabox.width - mediabox.left)
        height = abs(mediabox.height - mediabox.bottom)
        return width, height

    def _render_pdf_page_to_base64png(self, pdf_bytes: bytes, page_num: int, pdf_reader: PdfReader) -> str:
        """Render a PDF page to base64 PNG entirely in memory using pdftoppm."""
        width, height = self._get_pdf_page_dimensions(pdf_reader, page_num)
        longest_dim = max(width, height)
        dpi = int(self.target_longest_image_dim * 72 / longest_dim)

        # pdftoppm doesn't natively support stdin, so we use a shell wrapper
        # to pipe the PDF bytes through stdin and redirect to pdftoppm
        shell_cmd = f"pdftoppm -png -f {page_num} -l {page_num} -r {dpi} /dev/stdin"

        pdftoppm_result = subprocess.run(
            shell_cmd,
            shell=True,
            input=pdf_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )

        if pdftoppm_result.returncode != 0:
            raise RuntimeError(f"pdftoppm failed: {pdftoppm_result.stderr.decode()}. Ensure pdftoppm is installed and supports reading from /dev/stdin.")

        return base64.b64encode(pdftoppm_result.stdout).decode("utf-8")

    async def _build_page_query(self, pdf_bytes: bytes, page: int, pdf_reader: PdfReader, image_rotation: int = 0) -> dict:
        """Build a query for processing a single PDF page."""
        assert image_rotation in [0, 90, 180, 270], "Invalid image rotation"

        image_base64 = await asyncio.to_thread(self._render_pdf_page_to_base64png, pdf_bytes, page, pdf_reader)

        if image_rotation != 0:
            image_bytes = base64.b64decode(image_base64)
            with Image.open(BytesIO(image_bytes)) as img:
                if image_rotation == 90:
                    transpose = Image.Transpose.ROTATE_90
                elif image_rotation == 180:
                    transpose = Image.Transpose.ROTATE_180
                else:
                    transpose = Image.Transpose.ROTATE_270

                rotated_img = img.transpose(transpose)

                buffered = BytesIO()
                rotated_img.save(buffered, format="PNG")

            image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
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

    async def _process_page(self, pdf_bytes: bytes, page_num: int, pdf_reader: PdfReader) -> str:
        """Process a single page and return its text."""
        MODEL_MAX_CONTEXT = 16384
        TEMPERATURE_BY_ATTEMPT = [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]
        exponential_backoffs = 0
        cumulative_rotation = 0
        attempt = 0

        while attempt < self.max_retries:
            lookup_attempt = min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)

            query = await self._build_page_query(pdf_bytes, page_num, pdf_reader, image_rotation=cumulative_rotation)
            query["temperature"] = TEMPERATURE_BY_ATTEMPT[lookup_attempt]

            try:
                status_code, response_body = await self._apost(json_data=query)

                if status_code == 400:
                    raise ValueError(f"Got BadRequestError from server: {response_body}, skipping this response")
                elif status_code == 429:
                    raise ConnectionError("Too many requests, doing exponential backoff")
                elif status_code == 500:
                    raise ValueError(f"Got InternalServerError from server: {response_body}, skipping this response")
                elif status_code != 200:
                    raise ValueError(f"Error http status {status_code}")

                base_response_data = json.loads(response_body)

                if base_response_data["usage"]["total_tokens"] > MODEL_MAX_CONTEXT:
                    raise ValueError(f"Response exceeded model_max_context of {MODEL_MAX_CONTEXT}, cannot use this response")

                if base_response_data["choices"][0]["finish_reason"] != "stop":
                    raise ValueError("Response did not finish with reason code 'stop', cannot use this response")

                model_response_markdown = base_response_data["choices"][0]["message"]["content"]

                parser = FrontMatterParser(front_matter_class=PageResponse)
                front_matter, text = parser._extract_front_matter_and_text(model_response_markdown)
                page_response = parser._parse_front_matter(front_matter, text)

                if not page_response.is_rotation_valid and attempt < self.max_retries - 1:
                    cumulative_rotation = (cumulative_rotation + page_response.rotation_correction) % 360
                    raise ValueError(f"invalid_page rotation, retrying with rotation {cumulative_rotation}")

                return page_response.natural_text or ""

            except (ConnectionError, OSError, asyncio.TimeoutError):
                # Now we want to do exponential backoff, and not count this as an actual page retry
                # Page retries are supposed to be for fixing bad results from the model, but actual requests to vllm
                # are supposed to work. Probably this means that the server is just restarting
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

        return ""

    async def extract_text(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF bytes using olmOCR.
        All processing happens in memory without saving to disk.

        Args:
            pdf_bytes: The PDF file content as bytes

        Returns:
            Combined text from all pages in the PDF

        Example:
            >>> client = OlmOCRClient("http://localhost:30024/v1")
            >>> pdf_bytes = open("document.pdf", "rb").read()
            >>> text = await client.extract_text(pdf_bytes)
            >>> print(text)
        """
        # Create PdfReader once to avoid recreating it for each page
        pdf_reader = PdfReader(BytesIO(pdf_bytes))
        num_pages = pdf_reader.get_num_pages()

        # Process all pages concurrently
        tasks = [self._process_page(pdf_bytes, page_num, pdf_reader) for page_num in range(1, num_pages + 1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results, handling exceptions gracefully
        page_texts = []
        for result in results:
            if isinstance(result, Exception):
                # If a page fails, add empty string
                page_texts.append("")
            else:
                page_texts.append(result)

        # Combine all page texts
        return "\n".join(page_texts)

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
