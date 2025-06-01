//camera sht
const video = document.getElementById("C_feed")

navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
        video.play();
    })
    .catch((error) => {
        console.error("Error accessing the camera: ", error);
    });

const captureButton = document.getElementById("Capture_Button");

captureButton.addEventListener("click", () => {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
        const formData = new FormData();
        formData.append("image", blob, "capture.png");

        fetch("https://isrecyclable.netlify.app/predict", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                const previewDiv = document.getElementById("preview");
                if (data.prediction !== undefined) {
                    const classNames = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];
                    const recyclable = [true, true, true, true, true, false];
                    const className = classNames[data.prediction] || `Class ${data.prediction}`;
                    const isRecyclable = recyclable[data.prediction] ? "Recyclable" : "Not Recyclable";
                    previewDiv.innerHTML = `<b>Prediction:</b> ${className}<br><b>Status:</b> ${isRecyclable}`;
                } else if (data.error) {
                    previewDiv.innerText = `Error: ${data.error}`;
                } else {
                    previewDiv.innerText = "Unknown response from server.";
                }
            })
            .catch((error) => {
                const previewDiv = document.getElementById("preview");
                previewDiv.innerText = `Error sending image to the model: ${error}`;
            });
    }, "image/png");
});
