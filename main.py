import fitz
from transformers import pipeline


def open_document(pdf_path):
    return fitz.open(pdf_path)


def extract_text_from_pdf(document):
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text


def get_classes():
    return ['problem', 'method', 'goal', 'result', 'other']


def get_classifier():
    classifier = pipeline(
        'zero-shot-classification',
        model='facebook/bart-large-mnli',
    )
    return classifier


def classify_text(text, classifier, classes):
    chunks = text.split('\n')
    classifications = []
    for chunk in chunks:
        if chunk.strip():
            result = classifier(
                chunk,
                candidate_labels=classes,
            )
            label = result['labels'][0]
            if label == 'other':
                continue
            classifications.append((chunk, label))
    return classifications


def highlight_text_in_pdf(document, output_path, classified_text):
    color_map = {
        'problem': (1, 0, 0),  # red
        'method': (0, 1, 0),  # green
        'goal': (0, 0, 1),  # blue
        'result': (1, 1, 0),  # yellow
    }

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        for (text, label) in classified_text:
            areas = page.search_for(text)
            for area in areas:
                highlight = page.add_highlight_annot(area)
                highlight.set_colors(stroke=color_map[label.lower()])
                highlight.update()

    document.save(output_path)


def main():
    pdf_path = 'test.pdf'
    output_pdf_path = 'out_test.pdf'

    document = open_document(pdf_path)
    text = extract_text_from_pdf(document)
    labels = get_classes()
    classifier = get_classifier()
    classified_text = classify_text(text, classifier, labels)

    highlight_text_in_pdf(document, output_pdf_path, classified_text)


if __name__ == '__main__':
    main()
