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
        fetch("https://enviroapp.onrender.com/predict", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                const previewDiv = document.getElementById("preview");
                if (data.material !== undefined) {
                    const isRecyclable = data.recyclable ? "Recyclable" : "Not Recyclable";
                    previewDiv.innerHTML = `<b>Material:</b> ${data.material}<br><b>Status:</b> ${isRecyclable}`;
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
