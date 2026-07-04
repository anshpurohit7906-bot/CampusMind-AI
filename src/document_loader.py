from pypdf import PdfReader
import os


class DocumentLoader:
    def extract_text(self, file_path):
        """
        Reads a PDF file and returns all extracted text as a single string.
        """

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            raise ValueError(
                f"The file '{file_path}' is empty."
            )

        try:
            reader = PdfReader(file_path)

            pages = []

            for page in reader.pages:
                extracted = page.extract_text()

                if extracted:
                    pages.append(extracted)

            return "\n".join(pages)

        except Exception as e:
            raise RuntimeError(
                f"Failed to read PDF: {e}"
            )