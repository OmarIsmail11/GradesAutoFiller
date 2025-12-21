from src.Extractions.paperExtraction import *
from src.Extractions.tableExtraction import *
from src.Extractions.cellsExtraction import *

def getGradesSheet(imgPath):
    # Scanned Image
    paper = extractPaper(imgPath)

    # Extract Table
    table = extractTable(paper)

    tableCells = extra