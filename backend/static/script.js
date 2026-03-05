function detectDisease() {
    let fileInput = document.getElementById("imageInput");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please select an image first.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerHTML =
            "<h3>Prediction: " + data.prediction +
            "<br>Confidence: " + data.confidence + "%</h3>";
    });
}

function generateImage() {
    fetch("/generate")
    .then(response => response.json())
    .then(data => {
        document.getElementById("ganResult").innerHTML =
            "<img src='" + data.image_url + "' width='256'>";
    });
}