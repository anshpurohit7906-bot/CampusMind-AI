from src.document_loader import DocumentLoader

loader = DocumentLoader()

text = loader.extract_text("data/sample.pdf")

print(text[:1000])