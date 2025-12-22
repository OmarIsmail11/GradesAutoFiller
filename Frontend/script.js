// --- DOM ELEMENTS ---
const columnList = document.getElementById('column-list');
const addColBtn = document.getElementById('add-col-btn');

// Table Elements
const processBtn = document.getElementById('process-btn');
const loader = document.getElementById('loader');
const resultArea = document.getElementById('result-area');
const downloadBtn = document.getElementById('download-btn');

// Bubble Sheet Elements
const processBubbleBtn = document.getElementById('process-bubble-btn');
const loaderBubble = document.getElementById('loader-bubble');
const resultAreaBubble = document.getElementById('result-area-bubble');
const downloadBubbleBtn = document.getElementById('download-bubble-btn');

// Tabs
const tabBtns = document.querySelectorAll('.tab-btn');
const views = document.querySelectorAll('.view-section');

// State
let tableExcelBlob = null;
let bubbleExcelBlob = null;

// --- TAB SWITCHING ---
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all
        tabBtns.forEach(b => b.classList.remove('active'));
        views.forEach(v => {
            v.classList.remove('active');
            v.classList.add('hidden');
        });

        // Add active to clicked
        btn.classList.add('active');
        const tabId = btn.getAttribute('data-tab');
        const activeView = document.getElementById(tabId);
        activeView.classList.remove('hidden');
        activeView.classList.add('active');
    });
});

// --- TABLE RECOGNITION LOGIC ---
function addColumn() {
    const div = document.createElement('div');
    div.className = 'column-item';
    div.innerHTML = `
        <input type="text" placeholder="Col Name" class="col-name">
        <select class="col-type">
            <option value="Number">ID (Printed)</option>
            <option value="Written Number">Grade (Handwritten)</option>
            <option value="Arabic Name">Arabic Name</option>
            <option value="English Name">English Name</option>
            <option value="Symbol">Symbol / Checkmark</option>
        </select>
        <input type="number" placeholder="Len" class="col-len" style="width: 50px;">
        <button onclick="this.parentElement.remove()" style="color:red; background:none; border:none; cursor:pointer;">✕</button>
    `;
    columnList.appendChild(div);
}

// Initial column
addColumn();
addColBtn.addEventListener('click', addColumn);

processBtn.addEventListener('click', async () => {
    const imageInput = document.getElementById('image-input');
    if (!imageInput.files[0]) return alert("Please upload an image first.");

    // Prepare Config
    const columns = [];
    document.querySelectorAll('.column-item').forEach(item => {
        const len = item.querySelector('.col-len').value;
        columns.push({
            name: item.querySelector('.col-name').value || "Unnamed",
            type: item.querySelector('.col-type').value,
            len: len ? parseInt(len) : null
        });
    });

    // UI Feedback
    loader.classList.remove('hidden');
    resultArea.classList.add('hidden');
    processBtn.disabled = true;

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    formData.append('columns', JSON.stringify(columns));

    try {
        const response = await fetch('http://localhost:8000/process-sheet', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Processing failed");

        tableExcelBlob = await response.blob();
        loader.classList.add('hidden');
        resultArea.classList.remove('hidden');
    } catch (err) {
        alert(err.message);
        loader.classList.add('hidden');
    } finally {
        processBtn.disabled = false;
    }
});

downloadBtn.addEventListener('click', () => {
    downloadBlob(tableExcelBlob, "Table_Grades.xlsx");
});

// --- BUBBLE SHEET LOGIC ---
processBubbleBtn.addEventListener('click', async () => {
    const modelKeyInput = document.getElementById('model-key-input');
    const imagesInput = document.getElementById('bubble-images-input');

    if (!modelKeyInput.files[0]) return alert("Please upload the Model Answer Key (.txt).");
    if (imagesInput.files.length === 0) return alert("Please upload at least one student paper.");

    // UI Feedback
    loaderBubble.classList.remove('hidden');
    resultAreaBubble.classList.add('hidden');
    processBubbleBtn.disabled = true;

    const formData = new FormData();
    formData.append('model_answer', modelKeyInput.files[0]);
    
    // Append all selected images
    for (let i = 0; i < imagesInput.files.length; i++) {
        formData.append('paper_images', imagesInput.files[i]);
    }

    try {
        // ASSUMING ENDPOINT NAME IS /grade-bubbles
        const response = await fetch('http://localhost:8000/grade-bubbles', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Grading failed");

        bubbleExcelBlob = await response.blob();
        loaderBubble.classList.add('hidden');
        resultAreaBubble.classList.remove('hidden');
    } catch (err) {
        alert(err.message);
        loaderBubble.classList.add('hidden');
    } finally {
        processBubbleBtn.disabled = false;
    }
});

downloadBubbleBtn.addEventListener('click', () => {
    downloadBlob(bubbleExcelBlob, "Bubble_Sheet_Grades.xlsx");
});

// Helper
function downloadBlob(blob, filename) {
    if (!blob) return;
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
}