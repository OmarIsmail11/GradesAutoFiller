const formData = new FormData();
formData.append('image', selectedFile); // The .jpg file

const columns = [
    {name: 'Student ID', type: 'Number', len: 7},
    {name: 'Arabic Name', type: 'Arabic Name'},
    {name: 'Score', type: 'Written Number', len: 2}
];
formData.append('columns', JSON.stringify(columns));

fetch('http://localhost:5000/process-sheet', {
    method: 'POST',
    body: formData
})
.then(response => response.blob())
.then(blob => {
    // Logic to download the blob as an .xlsx file
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "results.xlsx";
    a.click();
});