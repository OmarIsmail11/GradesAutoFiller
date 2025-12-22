import numpy as np
import math
import os
from Module1GradesSheet.src.Recognition.Symbols.ClassicalSymbolRecognition import ClassicalSymbolRecognizer
from Module1GradesSheet.src.Recognition.Symbols.YOLOSymbolRecognition import YOLOSymbolRecognizer

class SymbolRecognizer:
    def __init__(self, method="YOLO"):
        self.method = method
        
        self.classical = ClassicalSymbolRecognizer()
        
        yolo_path = os.getenv("SYMBOL_MODEL_PATH", r"Module1GradesSheet/models/YOLO_SYMBOLS.pt")
        self.yolo = YOLOSymbolRecognizer(yolo_path)

    def predict(self, cell_img, override_method=None):
        """
        Returns the calculated numerical grade for the symbol.
        """
        method_to_use = override_method if override_method else self.method

        if method_to_use == "YOLO":
            if self.yolo.model is not None:
                return self.yolo.predict(cell_img)
            else:
                print("YOLO requested but model missing. Fallback to Classical.")
                return self.classical.predict(cell_img)
        
        elif method_to_use == "CLASSICAL":
            return self.classical.predict(cell_img)
            
        else:
            print(f"Unknown method {method_to_use}, defaulting to Classical.")
            return self.classical.predict(cell_img)

symbol_recognizer = SymbolRecognizer()