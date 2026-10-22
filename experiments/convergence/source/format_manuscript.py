"""Apply the final layout-only revision after regenerating the manuscript."""
from pathlib import Path
import argparse, re

ROOT = Path(__file__).resolve().parent.parent

def normalized_body(text):
    text = text.split('\\begin{document}', 1)[1]
    text = re.sub(r'^\\FloatBarrier\s*$', '', text, flags=re.M)
    text = re.sub(r'(\\begin\{(?:figure|table)\})\[[^\]]*\]', r'\1', text)
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    return re.sub(r'\s+', '', text)

def format_source(text):
    original = text
    barriers = text.count('\\FloatBarrier\n')
    if barriers == 4:
        text = text.replace('\\FloatBarrier\n', '', 3)
    elif barriers != 1:
        raise ValueError('Expected four original barriers or one final conclusion barrier')
    text = text.replace('\\begin{figure}[!htbp]', '\\begin{figure}[!tbp]')
    insertion = ('\n% Compact references without altering their text or font size.\n'
        '\\setlength{\\bibsep}{4pt plus 1pt minus 1pt}\n')
    anchor = '\\setlength{\\belowcaptionskip}{4pt}\n'
    if insertion not in text:
        if text.count(anchor) != 1:
            raise ValueError('Expected the existing caption-spacing setting')
        text = text.replace(anchor, anchor+insertion)
    assert normalized_body(text) == normalized_body(original)
    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manuscript', nargs='?', default=str(ROOT/'FROST_with_convergence.tex'))
    args = parser.parse_args()
    path = Path(args.manuscript)
    text = path.read_text(encoding='utf-8')
    updated = format_source(text)
    path.write_text(updated, encoding='utf-8')
    print('Applied layout revision; all manuscript content preserved.')
