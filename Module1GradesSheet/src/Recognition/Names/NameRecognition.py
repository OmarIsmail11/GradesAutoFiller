import easyocr
import numpy as np

class NameRecognizer:
    def __init__(self):
        self._readers = {}
        print("NameRecognizer initialized. Models will load on first use.")

    def _get_reader(self, lang):
        if lang not in self._readers:
            print(f"--- Loading EasyOCR model for language: {lang} ---")
            # gpu=True is used here as requested previously
            self._readers[lang] = easyocr.Reader([lang], gpu=True)
        return self._readers[lang]

    def recognize_student_name(self, cell_img, lang_input='en'):
        if cell_img is None:
            return ""

        reader = self._get_reader(lang_input)
        
        results = reader.readtext(cell_img, detail=1)

        if not results:
            return ""

        if lang_input == 'ar':
            results.sort(key=lambda x: x[0][1][0], reverse=True)
        else:
            results.sort(key=lambda x: x[0][0][0])

        name_segments = [res[1] for res in results]
        
        name_string = " ".join(name_segments)
        
        return name_string.strip()

name_recognizer = NameRecognizer()