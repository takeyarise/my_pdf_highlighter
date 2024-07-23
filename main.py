import fitz
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import torch
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


def get_except_classes():
    return ['other']


def get_classifier():
    # NOTE: I want to make it work lightweight on CPU.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classifier = pipeline(
        task='zero-shot-classification',
        model='facebook/bart-large-mnli',
        device=device,
    )
    return classifier


def classify_text(text, classifier, classes):
    # chunks = text.split('\n')
    chunks = sent_tokenize(text)
    classifications = []
    for chunk in chunks:
        if chunk.strip():
            result = classifier(
                chunk,
                candidate_labels=classes,
            )
            label = result['labels'][0]
            score = result['scores'][0]
            if label in get_except_classes():
                continue
            classifications.append((chunk, label, score))
    return classifications


def filter_text(classified_text, threshold):
    return list(filter(lambda x: x[2] > threshold, classified_text))


def highlight_text_in_pdf(document, output_path, classified_text):
    color_map = {
        'problem': (1, 0, 0),  # red
        'method': (0, 1, 0),  # green
        'goal': (0, 0, 1),  # blue
        'result': (1, 1, 0),  # yellow
    }  # FIXME: remove hardcoding

    for page_num in range(len(document)):
        page = document.load_page(page_num)
        for (text, label, *_) in classified_text:
            areas = page.search_for(text)
            for area in areas:
                highlight = page.add_highlight_annot(area)
                highlight.set_colors(stroke=color_map[label.lower()])
                highlight.update()

    document.save(output_path)


def main():
    # FIXME: remove hardcoding
    pdf_path = 'test.pdf'
    output_pdf_path = 'out_test.pdf'
    threshold = 0.5

    print('Opening document')
    document = open_document(pdf_path)
    print('Extracting text from document')
    text = extract_text_from_pdf(document)
    print('Classifying text')
    labels = get_classes()
    classifier = get_classifier()
    classified_text = classify_text(text, classifier, labels)
    print('Filtering text')
    classified_text = filter_text(classified_text, threshold)
    print('Highlighting text in document')
    highlight_text_in_pdf(document, output_pdf_path, classified_text)


if __name__ == '__main__':
    main()
