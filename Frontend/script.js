const columnList = document.getElementById('column-list');
const addColBtn = document.getElementById('add-col-btn');
const processBtn = document.getElementById('process-btn');
const loader = document.getElementById('loader');
const resultArea = document.getElementById('result-area');
const downloadBtn = document.getElementById('download-btn');

let excelBlob = null;

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

    // Prepare Column Configuration
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

        excelBlob = await response.blob();
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
    if (!excelBlob) return;
    const url = window.URL.createObjectURL(excelBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "Grading_Results.xlsx";
    document.body.appendChild(a);
    a.click();
    a.remove();
});