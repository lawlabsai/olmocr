"""
Simple Python client for olmOCR that processes PDF bytes and returns text.
All processing happens in memory without saving to disk.
"""

import asyncio
import base64
import json
import ssl
import subprocess
from io import BytesIO
from urllib.parse import urlparse

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

    async def _apost(self, json_data: dict) -> tuple[int, bytes]:
        """Simple async HTTP POST implementation."""
        parsed_url = urlparse(self._completion_url)
        host = parsed_url.hostname
        if parsed_url.scheme == "https":
            port = parsed_url.port or 443
            use_ssl = True
        else:
            port = parsed_url.port or 80
            use_ssl = False
        path = parsed_url.path or "/"

        writer = None
        try:
            if use_ssl:
                ssl_context = ssl.create_default_context()
                reader, writer = await asyncio.open_connection(host, port, ssl=ssl_context)
            else:
                reader, writer = await asyncio.open_connection(host, port)

            json_payload = json.dumps(json_data)

            headers = [
                f"POST {path} HTTP/1.1",
                f"Host: {host}",
                f"Content-Type: application/json",
                f"Content-Length: {len(json_payload)}",
            ]

            if self.api_key:
                headers.append(f"Authorization: Bearer {self.api_key}")

            headers.append("Connection: close")

            request = "\r\n".join(headers) + "\r\n\r\n" + json_payload
            writer.write(request.encode())
            await writer.drain()

            status_line = await reader.readline()
            if not status_line:
                raise ConnectionError("No response from server")
            status_parts = status_line.decode().strip().split(" ", 2)
            if len(status_parts) < 2:
                raise ValueError(f"Malformed status line: {status_line.decode().strip()}")
            status_code = int(status_parts[1])

            # Read headers
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                key, _, value = line.decode().partition(":")
                headers[key.strip().lower()] = value.strip()

            # Read response body
            if "content-length" in headers:
                body_length = int(headers["content-length"])
                response_body = await reader.readexactly(body_length)
            elif headers.get("transfer-encoding", "") == "chunked":
                chunks = []
                while True:
                    size_line = await reader.readline()
                    chunk_size = int(size_line.strip(), 16)

                    if chunk_size == 0:
                        await reader.readline()
                        break

                    chunk_data = await reader.readexactly(chunk_size)
                    chunks.append(chunk_data)
                    await reader.readline()

                response_body = b"".join(chunks)
            elif headers.get("connection", "") == "close":
                response_body = await reader.read()
            else:
                raise ConnectionError("Cannot determine response body length")

            return status_code, response_body
        finally:
            if writer is not None:
                try:
                    writer.close()
                    await writer.wait_closed()
                except:
                    pass

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
        cumulative_rotation = 0

        for attempt in range(self.max_retries):
            lookup_attempt = min(attempt, len(TEMPERATURE_BY_ATTEMPT) - 1)

            query = await self._build_page_query(pdf_bytes, page_num, pdf_reader, image_rotation=cumulative_rotation)
            query["temperature"] = TEMPERATURE_BY_ATTEMPT[lookup_attempt]

            try:
                status_code, response_body = await self._apost(json_data=query)

                if status_code == 400:
                    raise ValueError(f"BadRequestError from server: {response_body}")
                elif status_code == 429:
                    raise ConnectionError("Too many requests")
                elif status_code == 500:
                    raise ValueError(f"InternalServerError from server: {response_body}")
                elif status_code != 200:
                    raise ValueError(f"Error http status {status_code}")

                base_response_data = json.loads(response_body)

                if base_response_data["usage"]["total_tokens"] > MODEL_MAX_CONTEXT:
                    raise ValueError(f"Response exceeded model_max_context of {MODEL_MAX_CONTEXT}")

                if base_response_data["choices"][0]["finish_reason"] != "stop":
                    raise ValueError("Response did not finish with reason code 'stop'")

                model_response_markdown = base_response_data["choices"][0]["message"]["content"]

                parser = FrontMatterParser(front_matter_class=PageResponse)
                front_matter, text = parser._extract_front_matter_and_text(model_response_markdown)
                page_response = parser._parse_front_matter(front_matter, text)

                if not page_response.is_rotation_valid and attempt < self.max_retries - 1:
                    cumulative_rotation = (cumulative_rotation + page_response.rotation_correction) % 360
                    raise ValueError(f"Invalid page rotation, retrying with rotation {cumulative_rotation}")

                return page_response.natural_text or ""

            except (ConnectionError, ValueError):
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(1 * (attempt + 1))

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
